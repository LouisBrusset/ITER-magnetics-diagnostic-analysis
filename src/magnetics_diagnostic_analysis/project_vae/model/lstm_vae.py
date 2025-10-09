import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LengthAwareLSTMEncoder(nn.Module):
    """
    LSTM encoder that processes input sequences and produces latent representations.
    It uses convolutional layers to downsample the input sequence before feeding it to the LSTM.

    Architecture:
        - Three convolutional layers to downsample the input sequence.
        - An LSTM to capture temporal dependencies, handling variable lengths with packed sequences.
        - Two linear layers to produce the mean and log-variance of the latent variable distribution.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Dimensionality of the LSTM hidden states.
        latent_dim: Dimensionality of the latent space.
        num_layers: Number of LSTM layers.
    """
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            latent_dim: int, 
            num_layers: int
            ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, input_dim*2, kernel_size=(5,), stride=2, padding=2),
            nn.SELU(),
            nn.Conv1d(input_dim*2, input_dim*4, kernel_size=(5,), stride=2, padding=2),
            nn.SELU(),
            nn.Conv1d(input_dim*4, input_dim*8, kernel_size=(5,), stride=2, padding=2)
        )
        self.encoder_lstm = nn.LSTM(input_dim*8, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.encoder_linear_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_linear_logvar = nn.Linear(hidden_dim, latent_dim)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(
            self, 
            x_padded: torch.Tensor, 
            lengths: torch.Tensor, 
            hidden: tuple = None
            ) -> tuple[torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x_padded: Padded input sequences of shape (batch_size, seq_len, input_dim).
            lengths: Actual lengths of each sequence in the batch.
            hidden: Optional initial hidden and cell states for the LSTM. Useful in the truncated BPTT. If None, they are initialized to zeros.
        """
        batch_size, _, _ = x_padded.shape
        compressed_lengths = lengths.clone()
        for _ in range(3):  # number of conv layers
            compressed_lengths = torch.div(compressed_lengths + 1, 2, rounding_mode='floor')

        x_padded = self.conv(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        if hidden is None or hidden[0] is None or hidden[1] is None:
            h0 = torch.zeros(self.encoder_lstm.num_layers, batch_size, self.encoder_lstm.hidden_size, device=x_padded.device)
            c0 = torch.zeros(self.encoder_lstm.num_layers, batch_size, self.encoder_lstm.hidden_size, device=x_padded.device)
            hidden = (h0, c0)

        packed_input = pack_padded_sequence(x_padded, compressed_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder_lstm(packed_input)
        #output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        last_hidden = hidden[0][-1]
        mean = self.encoder_linear_mean(last_hidden)
        logvar = self.encoder_linear_logvar(last_hidden)
        return mean, logvar, hidden
    

class LengthAwareLSTMDecoder(nn.Module):
    """
    LSTM decoder that reconstructs the input sequence from the latent space.

    Architecture:
        - Projects the latent vector to initialize the hidden and cell states of the LSTM.
        - Uses an LSTM to generate sequences, handling variable lengths with packed sequences.
        - Applies linear layers and transposed convolutions to upsample and reconstruct the original input dimensions
        - Ensures the output sequence matches the original input length by trimming or padding as necessary.
    
    Args:
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Dimensionality of the LSTM hidden states.
        output_dim: Dimensionality of the output features (should match input_dim of the encoder).
        num_layers: Number of LSTM layers.
    """
    def __init__(
            self, 
            latent_dim: int, 
            hidden_dim: int, 
            output_dim: int, 
            num_layers: int
            ) -> None:
        assert hidden_dim % 8 == 0, "Hidden dimension must be divisible by 8."
        assert hidden_dim//8 > output_dim, "Hidden dimension too small for the given output dimension."
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4),
            nn.SELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * num_layers * 2)  # For hidden and cell states of each layer
        )
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, output_dim*8)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(output_dim*8, output_dim*4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SELU(),
            nn.ConvTranspose1d(output_dim*4, output_dim*2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SELU(),
            nn.ConvTranspose1d(output_dim*2, output_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        #self.dropout = nn.Dropout(p=0.5)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights using Xavier uniform for weights and zeros for biases.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(
            self, 
            z: torch.Tensor, 
            lengths: torch.Tensor, 
            hidden: tuple = None
            ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z: Latent vectors of shape (batch_size, latent_dim).
            lengths: Actual lengths of each sequence in the batch.
            hidden: Optional initial hidden and cell states for the LSTM. Useful in the truncated BPTT. If None, they are initialized from z.
        
        Returns:
            deconv_output: Reconstructed sequences of shape (batch_size, seq_len, output_dim).
            hidden_out: Final hidden and cell states from the LSTM.
        """
        batch_size, _ = z.shape

        if hidden is None or hidden[0] is None or hidden[1] is None:
            init_states = self.latent_projection(z)
            h0 = init_states[:, :self.hidden_dim * self.num_layers].reshape(self.num_layers, batch_size, self.hidden_dim)
            c0 = init_states[:, self.hidden_dim * self.num_layers:].reshape(self.num_layers, batch_size, self.hidden_dim)
            hidden = (h0, c0)

        compressed_lengths = lengths.clone()
        for _ in range(3):  # number of conv layers
            compressed_lengths = torch.div(compressed_lengths + 1, 2, rounding_mode='floor')
        max_compressed_len = torch.max(compressed_lengths).item()
        input_seq = torch.zeros(batch_size, max_compressed_len, self.hidden_dim, device=z.device)

        packed_input = pack_padded_sequence(input_seq, compressed_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden_out = self.decoder_lstm(packed_input, hidden)

        transformed_data = self.decoder_layers(packed_output.data)

        output_packed = torch.nn.utils.rnn.PackedSequence(
            data=transformed_data, 
            batch_sizes=packed_output.batch_sizes,
            sorted_indices=packed_output.sorted_indices,
            unsorted_indices=packed_output.unsorted_indices
        )
        output, _ = pad_packed_sequence(output_packed, batch_first=True)

        deconv_output = self.deconv(output.permute(0, 2, 1)).permute(0, 2, 1)

        max_len = torch.max(lengths).item()
        if deconv_output.size(1) > max_len:
            deconv_output = deconv_output[:, :max_len, :]
        elif deconv_output.size(1) < max_len:
            pad = torch.zeros(batch_size, max_len - deconv_output.size(1), self.output_dim, device=z.device)
            deconv_output = torch.cat([deconv_output, pad], dim=1)

        return deconv_output, hidden_out
    

class LSTMBetaVAE(nn.Module):
    """
    LSTM-based Variational Autoencoder (VAE) with length-aware mechanisms.
    Supports both full Backpropagation Through Time (BPTT) and truncated BPTT (t-BPTT) for training on long sequences.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Dimensionality of the LSTM hidden states.
        latent_dim: Dimensionality of the latent space.
        lstm_num_layers: Number of LSTM layers in both encoder and decoder.
        bptt_steps: If specified, enables truncated BPTT with the given number of steps. If None, full BPTT is used.

    Returns:
        reconstruction: Reconstructed sequences of shape (batch_size, seq_len, input_dim).
        z_mean: Mean of the latent variable distribution of shape (batch_size, latent_dim).
        z_logvar: Log-variance of the latent variable distribution of shape (batch_size, latent_dim).
    """
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            latent_dim: int, 
            lstm_num_layers: int, 
            bptt_steps: None | int = None
            ) -> None:
        super().__init__()
        self.encoder = LengthAwareLSTMEncoder(input_dim, hidden_dim, latent_dim, lstm_num_layers)
        self.decoder = LengthAwareLSTMDecoder(latent_dim, hidden_dim, input_dim, lstm_num_layers)
        self.bptt_steps = bptt_steps
        self.latent_dim = latent_dim

    def forward(
            self, 
            x: torch.Tensor, 
            lengths: torch.Tensor
            ) -> tuple[torch.Tensor]:
        """
        This function handles both full BPTT and truncated BPTT based on the bptt_steps parameter.
        If bptt_steps is None, it performs full BPTT. Otherwise, it performs truncated BPTT.
        """
        if self.bptt_steps is not None:
            return self._forward_pass_with_Truncated_BPTT(x, lengths)
        return self._forward_pass_with_Full_BPTT(x, lengths)

    def _forward_pass_with_Truncated_BPTT(
            self, 
            x: torch.Tensor, 
            lengths: torch.Tensor
            ) -> tuple[torch.Tensor]:
        """
        Forward pass with truncated backpropagation through time (t-BPTT).
        The input sequences are processed in segments of length bptt_steps.
        Hidden states are detached between segments to limit the computational graph size. It allows training on longer sequences without running out of memory.
        It handles variable-length sequences using the provided lengths.
        The tensor initialization and management of hidden states ensure that only active sequences are processed in each segment.

        Args:
            x: Padded input sequences of shape (batch_size, seq_len, input_dim).
            lengths: Actual lengths of each sequence in the batch.
        Returns:
            reconstruction: Reconstructed sequences of shape (batch_size, seq_len, input_dim).
            z_mean: Mean of the latent variable distribution of shape (batch_size, latent_dim).
            z_logvar: Log-variance of the latent variable distribution of shape (batch_size, latent_dim).
        """
        batch_size, seq_len, _ = x.shape
        reconstructions = []
        all_z_means = []
        all_z_logvars = []
        segment_weights = []
        
        # Initialization of hidden states for encoder and decoder
        h_enc, c_enc = None, None
        h_dec, c_dec = None, None
        
        # Loop over the partition of time dimension
        for start_idx in range(0, seq_len, self.bptt_steps):
            end_idx = min(start_idx + self.bptt_steps, seq_len)
            segment_x = x[:, start_idx:end_idx, :]
            segment_lengths = (lengths - start_idx).clamp(min=0, max=self.bptt_steps)

            weights = segment_lengths.float() / segment_lengths.max().clamp(min=1)
            segment_weights.append(weights)

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

        weights_tensor = torch.stack(segment_weights, dim=1)  # [batch, segments]
        weights_tensor = weights_tensor.unsqueeze(-1)         # [batch, segments, 1]

        z_means_all = torch.stack(all_z_means, dim=1)         # [batch, segments, latent_dim]
        z_logvars_all = torch.stack(all_z_logvars, dim=1)
        z_mean = torch.sum(z_means_all * weights_tensor, dim=1) / torch.sum(weights_tensor, dim=1)
        z_logvar = torch.sum(z_logvars_all * weights_tensor, dim=1) / torch.sum(weights_tensor, dim=1)
        reconstruction = torch.cat(reconstructions, dim=1)
        
        return reconstruction, z_mean, z_logvar

    def _forward_pass_with_Full_BPTT(
            self, 
            x: torch.Tensor, 
            lengths: torch.Tensor
            ) -> tuple[torch.Tensor]:
        """
        Forward pass with full backpropagation through time (BPTT).
        It handles variable-length sequences using the provided lengths.
        The entire sequence is processed in one go, allowing gradients to flow through the entire sequence.

        Args:
            x: Padded input sequences of shape (batch_size, seq_len, input_dim).
            lengths: Actual lengths of each sequence in the batch.

        Returns:
            reconstruction: Reconstructed sequences of shape (batch_size, seq_len, input_dim).
            z_mean: Mean of the latent variable distribution of shape (batch_size, latent_dim).
            z_logvar: Log-variance of the latent variable distribution of shape (batch_size,
        """
        z_mean, z_logvar, _ = self.encoder(x, lengths)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed, _ = self.decoder(z, lengths)
        return x_reconstructed, z_mean, z_logvar

    def reparameterize(
            self, 
            mean: torch.Tensor, 
            logvar: torch.Tensor
            ) -> torch.Tensor:
        """Reparameterization trick to sample from N(mean, var) to create a latent variable."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std