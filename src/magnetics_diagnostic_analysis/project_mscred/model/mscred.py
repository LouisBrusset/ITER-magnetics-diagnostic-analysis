import torch
import torch.nn as nn

from collections import deque

from magnetics_diagnostic_analysis.project_mscred.model.convlstm import ConvLSTM


## Attention mechanism for Conv_LSTM output
def attention(
        ConvLstm_out: torch.Tensor,
        rescale_factor: float = 5.0
        ) -> torch.Tensor:
    """
    Attention mechanism for ConvLSTM output (Vectorized)
    This implementation build a causal "by construction" attention mechanism. "By construction" because we only give passed informations.

    Args:
        ConvLstm_out: Tensor output from the ConvLSTM layer. Shape (n_effective_timestep, channels, height, width)
        n: Number of steps to consider for attention weighting.

    Returns:
        Tensor: The output after applying the attention mechanism. Shape (channels, height, width)

    Nota bene:
        The rescale_factor is used to normalize the similarity scores after the dot product of the 
        hidden states and before the softmax function.
        The value of 5 is empirically determined.
        This value is a compromise between producing uniform attention weights (large rescale_factor = all weights are the same order) 
        and producing a sharp distribution in weights (small rescale_factor = one weight dominates) with the softmax function.
    
    Post Scriptum: 
        In the future, we could replace this attention mechanism with a more sophisticated one, such as a query-key-value attention mechanism.
    """
    assert rescale_factor > 0, "Rescale factor must be positive and non zero"

    last_step = ConvLstm_out[-1]
    similarities = torch.einsum('tchw,chw->t', ConvLstm_out, last_step) / rescale_factor
    attention_weights = torch.softmax(similarities, dim=0)
    ConvLstm_out_weighted = torch.einsum('t,tchw->chw', attention_weights, ConvLstm_out)
    return ConvLstm_out_weighted


## Encoder-Decoder modules
class CnnEncoder(nn.Module):
    """
    CNN Encoder module

    This module applies a series of convolutional layers to the input data.
    Shapes: (batch_size, 32,  height,    width)    after conv1 (stride 1)
            (batch_size, 64,  height//2, width//2) after conv2 (stride 2)
            (batch_size, 128, height//4, width//4) after conv3 (stride 2)
            (batch_size, 256, height//8, width//8) after conv4 (stride 2)
    """
    def __init__(
            self, 
            in_channels_encoder: int = 3, 
            deep_channel_sizes: list[int] = [32, 64, 128, 256]
            ) -> None:
        """
        We have chosen SELU() activation function for its self-normalizing properties.

        Advices:
            - in_channels = [in_channels_encoder, 32, 64, 128]
            - out_channels = [32, 64, 128, 256]
            - kernel_sizes = [3, 3, 2, 2]
            - stride must be (2, 2) for the first three layers, (1, 1) for the last layer (no dimension reduction)
            - padding must be (0, 0) for the last layer (no operations after the last layer), (1, 1) for the others
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_encoder, 
                      out_channels=deep_channel_sizes[0], 
                      kernel_size=3, stride=(1, 1), padding=1),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=deep_channel_sizes[0], 
                      out_channels=deep_channel_sizes[1], 
                      kernel_size=3, stride=(2, 2), padding=1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=deep_channel_sizes[1], 
                      out_channels=deep_channel_sizes[2], 
                      kernel_size=2, stride=(2, 2), padding=0),
            nn.SELU()
        )   
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=deep_channel_sizes[2], 
        #               out_channels=deep_channel_sizes[3], 
        #               kernel_size=2, stride=(2, 2), padding=0),
        #     nn.SELU()
        # )
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Forward pass for the CNN encoder

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            tuple[torch.Tensor]: Output tensors from each convolutional layer
        """
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out#, conv4_out

class Conv_LSTM(nn.Module):
    """
    Convolutional LSTM module
    This module applies a series of ConvLSTM layers to the output of CnnEncoder.

    Functions:
        - __init__: Initializes the Conv_LSTM module.
        - forward: Defines the forward pass for the Conv_LSTM module.
        - _init_queue: Initializes the internal queues for the Conv_LSTM module.
        - _update_queues: Updates the internal queues with the latest CnnEncoder output.
    """
    def __init__(
            self,
            deep_channel_sizes: list[int] = [32, 64, 128, 256],
            num_layers: int = 1,
            n_timesteps: int = 5, 
            n_effective_timesteps: list = [1, 2, 3, 4]
            ) -> None:
        super().__init__()
        self.conv1_lstm = ConvLSTM(input_channels=deep_channel_sizes[0], 
                                   hidden_channels=[deep_channel_sizes[0]]*num_layers, 
                                   kernel_size=3, 
                                   step=n_timesteps, effective_step=n_effective_timesteps)
        self.conv2_lstm = ConvLSTM(input_channels=deep_channel_sizes[1], 
                                   hidden_channels=[deep_channel_sizes[1]]*num_layers, 
                                   kernel_size=3, 
                                   step=n_timesteps, effective_step=n_effective_timesteps)
        self.conv3_lstm = ConvLSTM(input_channels=deep_channel_sizes[2], 
                                   hidden_channels=[deep_channel_sizes[2]]*num_layers, 
                                   kernel_size=3, 
                                   step=n_timesteps, effective_step=n_effective_timesteps)
        # self.conv4_lstm = ConvLSTM(input_channels=deep_channel_sizes[3], 
        #                            hidden_channels=[deep_channel_sizes[3]]*num_layers, 
        #                            kernel_size=3, 
        #                            step=n_timesteps, effective_step=n_effective_timesteps)

        self.n_timesteps = n_timesteps
        self.queue1 = deque(maxlen=n_timesteps)
        self.queue2 = deque(maxlen=n_timesteps)
        self.queue3 = deque(maxlen=n_timesteps)
        # self.queue4 = deque(maxlen=n_timestep)

    def _init_queue(
            self, 
            queue: deque, 
            in_shape: tuple, 
            steps: int,
            device: str
            ) -> None:
        """
        Initialize the queues with zeros for the first `steps - 1` timesteps.

        Args:
            queue (deque): Queue to initialize
            in_shape (tuple): Shape of the input tensor
            steps (int): Number of timesteps
            device (str): Device to place the tensors on

        Nota bene:
            The zeros tensors are created with `requires_grad=False` to avoid unnecessary gradient computations.
        """
        queue.clear()
        for _ in range(steps):
            # resize to be shape [steps, C, H, W]
            queue.append(torch.zeros(in_shape[1:], device=device, requires_grad=False))

    def _update_queues(self, x_conv1_out: torch.Tensor, x_conv2_out: torch.Tensor, x_conv3_out: torch.Tensor, x_conv4_out: torch.Tensor = 0) -> None:
        """
        Update the queues with the latest CnnEncoder output with this scheme:
            - Theoretically, in the last queue, all tensors are detached form autograd, except for the most recent one.
            - We pop the recent tensors from the queues and detach them from the computation graph.
            - We add this detached tensor back to the queue.
            - We add the new tensor to the queue.
            - This new tensor is attached to the computation graph.
            - Thanks to maxlen deque's behavior, we ensure that the queue always contains the n_timesteps most recent tensors.

        Args:
            x_convi_out (torch.Tensor): Output from the i_th CnnEncoder layer.
        """
        # resize to be shape [steps, C, H, W]
        assert len(self.queue1) == self.n_timesteps, "Queue length must match n_timesteps"
        assert len(self.queue2) == self.n_timesteps, "Queue length must match n_timesteps"
        assert len(self.queue3) == self.n_timesteps, "Queue length must match n_timesteps"
        # assert len(self.queue4) == self.n_timesteps, "Queue length must match n_timesteps"

        old_grad_tensor1 = self.queue1.pop()
        old_grad_tensor2 = self.queue2.pop()
        old_grad_tensor3 = self.queue3.pop()
        # old_grad_tensor4 = self.queue4.pop()
        
        old_grad_tensor1 = old_grad_tensor1.detach()
        old_grad_tensor2 = old_grad_tensor2.detach()
        old_grad_tensor3 = old_grad_tensor3.detach()
        # old_grad_tensor4 = old_grad_tensor4.detach()

        self.queue1.append(old_grad_tensor1)
        self.queue2.append(old_grad_tensor2)
        self.queue3.append(old_grad_tensor3)
        # self.queue4.append(old_tensor4)

        self.queue1.append(x_conv1_out)
        self.queue2.append(x_conv2_out)
        self.queue3.append(x_conv3_out)
        # self.queue4.append(x_conv4_out)

    def forward(self, conv1_out: torch.Tensor, conv2_out: torch.Tensor, conv3_out: torch.Tensor, conv4_out: torch.Tensor = 0) -> tuple[torch.Tensor]:
        """
        ConvLSTM forward pass
        convi_lstm_out is a tuple (outputs, (x, new_c))
            - output: the registered hidden state of effective_step list
            - (x, new_c): the hidden state and cell state for the last step
        Exemple of shape for conv2_lstm_out:
            output.shape = (batch_size, 64, height/2, width/2)
            x.shape = new_c.shape = (batch_size, 64, height/2, width/2)
        So convi_lstm_out[0][0] is the first element of output.

        Args:
            convi_out (torch.Tensor): Output from the ith convolutional layer
            conv4_out (torch.Tensor, optional): Output from the fourth convolutional layer

        Returns:
            Tuple[torch.Tensor]: Output tensors from each ConvLSTM layer
        """
        for queue, convX in [(self.queue1, conv1_out), (self.queue2, conv2_out), (self.queue3, conv3_out)]:
            if bool(queue) is False:
                self._init_queue(queue, convX.shape, self.n_timesteps, device=convX.device)

        conv1_lstm_out = torch.zeros_like(conv1_out).to(device=conv1_out.device)
        conv2_lstm_out = torch.zeros_like(conv2_out).to(device=conv2_out.device)
        conv3_lstm_out = torch.zeros_like(conv3_out).to(device=conv3_out.device)
        # conv4_lstm_out = torch.zeros_like(conv4_out).to(device=conv4_out.device)

        for batch_idx, (x_conv1, x_conv2, x_conv3) in enumerate(zip(conv1_out, conv2_out, conv3_out)):
            x_conv1 = x_conv1.to(conv1_out.device)
            x_conv2 = x_conv2.to(conv2_out.device)
            x_conv3 = x_conv3.to(conv3_out.device)
            # x_conv4 = x_conv4.to(conv4_out.device)

            self._update_queues(x_conv1, x_conv2, x_conv3)

            x_conv1_lstm_in = torch.stack(tuple(self.queue1), dim=0)
            x_conv2_lstm_in = torch.stack(tuple(self.queue2), dim=0)
            x_conv3_lstm_in = torch.stack(tuple(self.queue3), dim=0)
            # x_conv4_lstm_in = torch.stack(tuple(self.queue4), dim=0)

            x_conv1_lstm_out = self.conv1_lstm(x_conv1_lstm_in)
            x_conv1_lstm_out = attention(ConvLstm_out=torch.cat(x_conv1_lstm_out[0], dim=0))
            x_conv2_lstm_out = self.conv2_lstm(x_conv2_lstm_in)
            x_conv2_lstm_out = attention(ConvLstm_out=torch.cat(x_conv2_lstm_out[0], dim=0))
            x_conv3_lstm_out = self.conv3_lstm(x_conv3_lstm_in)
            x_conv3_lstm_out = attention(ConvLstm_out=torch.cat(x_conv3_lstm_out[0], dim=0))
            # x_conv4_lstm_out = self.conv4_lstm(x_conv4_lstm_in)
            # x_conv4_lstm_out = attention(ConvLstm_out=torch.cat(x_conv4_lstm_out[0], dim=0))
            
            conv1_lstm_out[batch_idx] = x_conv1_lstm_out
            conv2_lstm_out[batch_idx] = x_conv2_lstm_out
            conv3_lstm_out[batch_idx] = x_conv3_lstm_out
            # conv4_lstm_out[batch_idx] = x_conv4_lstm_out

        return conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, #conv4_lstm_out

class CnnDecoder(nn.Module):
    """
    CNN Decoder module

    This module applies a series of transposed convolutional layers to the input data.
    Shapes: (batch_size, 128, height*2, width*2) after deconv4 (stride 2)
            (batch_size, 256, height*2, width*2) after first concatenation
            (batch_size, 64,  height*4, width*4) after deconv3 (stride 2)
            (batch_size, 128, height*4, width*4) after second concatenation
            (batch_size, 32,  height*8, width*8) after deconv2 (stride 2)
            (batch_size, 64,  height*8, width*8) after first concatenation
            (batch_size, 3,   height*8, width*8) after deconv1 (stride 1)
    """
    def __init__(
            self,
            in_channels_encoder: int = 3,
            deep_channel_sizes: list[int] = [32, 64, 128, 256]
            ) -> None:
        """
        We have chosen SELU() activation function for its self-normalizing properties.
        
        Advices:
            - in_channels = hidden_channels of the corresponding ConvLSTM layer + concatenation
                + if conv4_lstm_out is not None, [256, 256, 128, 64]
                + if conv4_lstm_out is None, [128, 128, 64]
            - out_channels
                + if conv4_lstm_out is not None, [128, 64, 32, encoder_in_channels]
                + if conv4_lstm_out is None, [64, 32, encoder_in_channels]
            - kernel_sizes = [2, 2, 3, 3]
            - stride must be (2, 2) for the first three layers, (1, 1) for the last layer (no dimension reduction)
            - padding must be (0, 0) for the first layer, (1, 1) for the others
            - output_padding must be (0, 0) for the first and last layer, (1, 1) for the others
        """
        super().__init__()
        deep_channel_sizes_decode = deep_channel_sizes[::-1]

        # self.deconv4 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0),
        #     nn.SELU()
        # )
        ### If conv4_lstm_out is not None, put in_channels(self.deconv3)=256, else 128
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=deep_channel_sizes_decode[0], 
                               out_channels=deep_channel_sizes_decode[1], 
                               kernel_size=2, stride=2, 
                               padding=0, output_padding=0),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=deep_channel_sizes_decode[1]*2, 
                               out_channels=deep_channel_sizes_decode[2], 
                               kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=deep_channel_sizes_decode[2]*2, 
                               out_channels=in_channels_encoder, 
                               kernel_size=3, stride=1, 
                               padding=1, output_padding=0),
            nn.SELU()
        )

    def forward(self, conv1_lstm_out: torch.Tensor, conv2_lstm_out: torch.Tensor, conv3_lstm_out: torch.Tensor, conv4_lstm_out: torch.Tensor = 0) -> torch.Tensor:
        """
        Forward pass for the CNN decoder

        Args:
            conv1_lstm_out (torch.Tensor): Output from the first LSTM layer
            conv2_lstm_out (torch.Tensor): Output from the second LSTM layer
            conv3_lstm_out (torch.Tensor): Output from the third LSTM layer
            conv4_lstm_out (torch.Tensor, optional): Output from the fourth LSTM layer
        Shapes:
            - conv1_lstm_out: (batch_size, 32, height, width)
            - conv2_lstm_out: (batch_size, 64, height, width)
            - conv3_lstm_out: (batch_size, 128, height, width)
            - conv4_lstm_out: (batch_size, 256, height, width)

        Returns:
            torch.Tensor: Output tensor from the decoder. Shape (batch_size, in_channels_encoder, height, width)
        """
        # deconv4 = self.deconv4(conv4_lstm_out)
        # deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim = 1)
        deconv3 = self.deconv3(conv3_lstm_out)      #deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim = 1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim = 1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1
    


## Encoder-Decoder global model
class MSCRED(nn.Module):
    """
    Multi-Scale Convolutional Recurrent Encoder-Decoder

    This model combines CNN Autoencoder and ConvLSTM layers for spatiotemporal feature extraction.
    """
    def __init__(
            self, 
            encoder_in_channel: int, 
            deep_channel_sizes: list[int], 
            lstm_num_layers: int = 1,
            lstm_timesteps: int = 5, 
            lstm_effective_timesteps: list[int] | str = 'all'
            ) -> None:
        super().__init__()
        assert len(lstm_effective_timesteps) > 0 and len(lstm_effective_timesteps) <= lstm_timesteps, "Effective timesteps must be non-empty and less than or equal to total timesteps"

        if lstm_effective_timesteps == 'all':
            lstm_effective_timesteps = list(range(lstm_timesteps))

        self.cnn_encoder = CnnEncoder(in_channels_encoder=encoder_in_channel, deep_channel_sizes=deep_channel_sizes)
        self.conv_lstm = Conv_LSTM(deep_channel_sizes=deep_channel_sizes, num_layers=lstm_num_layers, n_timesteps=lstm_timesteps, n_effective_timesteps=lstm_effective_timesteps)
        self.cnn_decoder = CnnDecoder(in_channels_encoder=encoder_in_channel, deep_channel_sizes=deep_channel_sizes)

        self.time_steps = lstm_timesteps
        self.model_depth = len(deep_channel_sizes)

    def reset_lstm_hidden_states(self) -> None:
        """
        Initializes all the hidden states (h,c) of the ConvLSTM layers.
        """
        if hasattr(self, 'conv_lstm'):
            for i in range(self.model_depth):
                if hasattr(self.conv_lstm, f'conv{i+1}_lstm'):
                    getattr(self.conv_lstm, f'conv{i+1}_lstm').reset_hidden_state()
                else:
                    raise AttributeError(f"ConvLSTM layer conv{i+1}_lstm does not exist in Conv_LSTM module.")
        else:
            raise AttributeError("ConvLSTM module does not exist.")

    # def forward(self, x):
    #     # Use gradient checkpointing to save GPU memory
    #     from torch.utils.checkpoint import checkpoint
    #     return checkpoint(self._forward, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MSCRED model

        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Reconstructed tensor from the decoder
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        # assert x.shape[0] > self.time_steps, "Input batch size must be greater than the number of time steps"
        # assert x.shape[0] % self.time_steps == 0, "Input batch size must be divisible by the number of time steps"

        conv1_out, conv2_out, conv3_out = self.cnn_encoder(x)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out = self.conv_lstm(conv1_out, conv2_out, conv3_out)
        gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out)
        
        return gen_x
