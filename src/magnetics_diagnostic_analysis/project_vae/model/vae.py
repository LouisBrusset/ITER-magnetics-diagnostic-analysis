import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LengthAwareLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.encoder_linear_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_linear_logvar = nn.Linear(hidden_dim, latent_dim)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor, hidden: tuple = None) -> tuple[torch.Tensor]:
        batch_size, _, _ = x_padded.shape
        if hidden is None or hidden[0] is None or hidden[1] is None:
            h0 = torch.zeros(self.encoder_lstm.num_layers, batch_size, self.encoder_lstm.hidden_size, device=x_padded.device)
            c0 = torch.zeros(self.encoder_lstm.num_layers, batch_size, self.encoder_lstm.hidden_size, device=x_padded.device)
            hidden = (h0, c0)

        packed_input = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder_lstm(packed_input)
        #output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        last_hidden = hidden[0][-1]
        mean = self.encoder_linear_mean(last_hidden)
        logvar = self.encoder_linear_logvar(last_hidden)
        return mean, logvar, hidden
    

class LengthAwareLSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.decoder_linear_init = nn.Linear(latent_dim, hidden_dim * num_layers * 2)  # For hidden and cell states of each layer
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.decoder_output_layer = nn.Linear(hidden_dim, output_dim)
        #self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, z: torch.Tensor, lengths: torch.Tensor, hidden: tuple = None) -> torch.Tensor:
        batch_size, _ = z.shape
        if hidden is None or hidden[0] is None or hidden[1] is None:
            init_states = self.decoder_linear_init(z)
            h0 = init_states[:, :self.hidden_dim * self.num_layers].reshape(self.num_layers, batch_size, self.hidden_dim)
            c0 = init_states[:, self.hidden_dim * self.num_layers:].reshape(self.num_layers, batch_size, self.hidden_dim)
            hidden = (h0, c0)

        max_length = torch.max(lengths)
        input_seq = torch.zeros(batch_size, max_length, self.hidden_dim, device=z.device)

        packed_input = pack_padded_sequence(input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden_out = self.decoder_lstm(packed_input, hidden)

        transformed_data = self.decoder_output_layer(packed_output.data)
        output_packed = torch.nn.utils.rnn.PackedSequence(
            data=transformed_data, 
            batch_sizes=packed_output.batch_sizes,
            sorted_indices=packed_output.sorted_indices,
            unsorted_indices=packed_output.unsorted_indices
        )
        output, _ = pad_packed_sequence(output_packed, batch_first=True)

        return output, hidden_out
    

class LSTMBetaVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lstm_num_layers: int, bptt_steps: None | int = None) -> None:
        super().__init__()
        self.encoder = LengthAwareLSTMEncoder(input_dim, hidden_dim, latent_dim, lstm_num_layers)
        self.decoder = LengthAwareLSTMDecoder(latent_dim, hidden_dim, input_dim, lstm_num_layers)
        self.bptt_steps = bptt_steps
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor]:
        if self.bptt_steps is not None:
            return self._forward_pass_with_Truncated_BPTT(x, lengths)
        return self._forward_pass_with_Full_BPTT(x, lengths)

    def _forward_pass_with_Truncated_BPTT(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        reconstructions = []
        all_z_means = []
        all_z_logvars = []
        
        # Initialization of hidden states for encoder and decoder
        h_enc, c_enc = None, None
        h_dec, c_dec = None, None
        
        # Loop over the partition of time dimension
        for start_idx in range(0, seq_len, self.bptt_steps):
            end_idx = min(start_idx + self.bptt_steps, seq_len)
            segment_x = x[:, start_idx:end_idx, :]
            segment_lengths = (lengths - start_idx).clamp(min=0, max=self.bptt_steps)

            # Create a mask for sequences that are still active in this segment
            active_indices = torch.where(segment_lengths > 0)[0]
            num_active = len(active_indices)

            # MAIN IDEA: Detach hidden states between segments to lighten the computational graph for gradient calculation.
            if h_enc is not None:
                h_enc, c_enc = h_enc.detach(), c_enc.detach()
            if h_dec is not None:
                h_dec, c_dec = h_dec.detach(), c_dec.detach()

            # Only process if there are active sequences in this segment
            if num_active > 0:
                # Extract only active sequences
                active_segment_x = segment_x[active_indices]
                active_segment_lengths = segment_lengths[active_indices]

                # Extract hidden states for active sequences only
                if h_enc is not None:
                    active_h_enc = h_enc[:, active_indices, :]
                    active_c_enc = c_enc[:, active_indices, :]
                else:
                    active_h_enc, active_c_enc = None, None

                # ENCODER
                z_mean_segment, z_logvar_segment, (new_h_enc, new_c_enc) = self.encoder(active_segment_x, active_segment_lengths, (active_h_enc, active_c_enc))

                # Update hidden states for active sequences
                if h_enc is None:
                    h_enc = torch.zeros(self.encoder.encoder_lstm.num_layers, batch_size, 
                                      self.encoder.encoder_lstm.hidden_size, device=x.device)
                    c_enc = torch.zeros_like(h_enc)
                
                h_enc[:, active_indices, :] = new_h_enc
                c_enc[:, active_indices, :] = new_c_enc


                # DECODER
                z_segment = self.reparameterize(z_mean_segment, z_logvar_segment)

                # Extract decoder hidden states for active sequences
                if h_dec is not None:
                    active_h_dec = h_dec[:, active_indices, :]
                    active_c_dec = c_dec[:, active_indices, :]
                else:
                    active_h_dec, active_c_dec = None, None
                
                x_reconstructed, (new_h_dec, new_c_dec) = self.decoder(z_segment, active_segment_lengths, (active_h_dec, active_c_dec))

                # Update decoder hidden states for active sequences
                if h_dec is None:
                    h_dec = torch.zeros(self.decoder.decoder_lstm.num_layers, batch_size, 
                                      self.decoder.decoder_lstm.hidden_size, device=x.device)
                    c_dec = torch.zeros_like(h_dec)

                h_dec[:, active_indices, :] = new_h_dec
                c_dec[:, active_indices, :] = new_c_dec

                # Create full reconstruction tensor with zeros for inactive sequences
                full_reconstruction = torch.zeros_like(segment_x)
                full_reconstruction[active_indices] = x_reconstructed
                
                # Create full latent tensors with zeros for inactive sequences
                full_z_mean = torch.zeros(batch_size, self.latent_dim, device=x.device)
                full_z_logvar = torch.zeros(batch_size, self.latent_dim, device=x.device)
                full_z_mean[active_indices] = z_mean_segment
                full_z_logvar[active_indices] = z_logvar_segment
            else:
                # No active sequences in this segment
                full_reconstruction = torch.zeros_like(segment_x)
                full_z_mean = torch.zeros(batch_size, self.latent_dim, device=x.device)
                full_z_logvar = torch.zeros(batch_size, self.latent_dim, device=x.device)

            reconstructions.append(full_reconstruction)
            all_z_means.append(full_z_mean)
            all_z_logvars.append(full_z_logvar)

        z_means_all = torch.stack(all_z_means, dim=1)  # [batch, segments, latent_dim]
        z_logvars_all = torch.stack(all_z_logvars, dim=1)
        z_mean = torch.mean(z_means_all, dim=1)
        z_logvar = torch.mean(z_logvars_all, dim=1)
        reconstruction = torch.cat(reconstructions, dim=1)
        
        return reconstruction, z_mean, z_logvar

    def _forward_pass_with_Full_BPTT(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor]:
        z_mean, z_logvar, _ = self.encoder(x, lengths)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed, _ = self.decoder(z, lengths)
        return x_reconstructed, z_mean, z_logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std