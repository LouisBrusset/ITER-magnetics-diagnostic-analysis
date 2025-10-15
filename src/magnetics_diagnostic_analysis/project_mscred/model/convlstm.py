import torch
import torch.nn as nn


## Convolutional Long-Short Term Memory (ConvLSTM) Cell
class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for processing spatio-temporal data.
    The discrepancy between the ConvLSTM and simple LSTM lies in the use of convolutional operations instead of fully connected layers,
    allowing the ConvLSTM to capture spatial hierarchies in the data.

    Methods:
        __init__: Initializes the ConvLSTM cell with input channels, hidden channels, and kernel size.
        forward: Performs the forward pass of the ConvLSTM cell.
        init_hidden: Initializes the hidden and cell states for the ConvLSTM cell.

    Attributes:
        input_channels: Number of input channels.
        hidden_channels: Number of hidden channels in the hidden state.
        kernel_size: Size of the convolutional kernel in the LSTM workflow.
        num_features: Number of features in the ConvLSTM cell (4 for input, forget, cell, output gates).
        padding: Padding size for the convolutional layers. The padding is calculated based on the kernel size.
        Wxi, Whi, Wxf, Whf, Wxc, Whc, Wxo, Who: Convolutional layers for input, forget, and output gates regarding the input and the hidden states.
        Wci, Wcf, Wco: Cell state weights for input, forget, and output gates regarding the cell state(initialized to None).
    """

    def __init__(
        self, input_channels: int, hidden_channels: int, kernel_size: int | tuple
    ) -> None:
        super().__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        # self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """
        Forward pass of the ConvLSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            h (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_channels, height, width)
            c (torch.Tensor): Cell state tensor of shape (batch_size, hidden_channels, height, width)

        Returns:
            Tuple of updated hidden state and cell state tensors, shape (batch_size, hidden_channels, height, width)
        """
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(
        self, batch_size: int, hidden: int, shape: tuple[int], device: str
    ) -> tuple[torch.Tensor]:
        """
        Initializes weight for the convLSTM cell. In two parts:
            - The weight matrices for the Input, Forget, and Output gates regarding the cell state (Wci, Wcf, Wco).
              These are initialized to zero tensors, in order to limit the influence of the cell state (and its gradient) in first iterations.

            - The very first hidden state and cell state (h and c). h and c represent respectively the short and long-term memory.
              Those are also initialized to zero tensors, because no history is available at the first time step.

        This is done only once, at the first step of the forward pass.

        Args:
            batch_size (int): Size of the batch
            hidden (int): Number of hidden channels
            shape (tuple[int]): Shape of the input (height, width)
            device (str): Device to place the tensors on

        Returns:
            Tuple of initialized hidden state and cell state tensors.
        """
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1], device=device)
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1], device=device)
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1], device=device)
        else:
            assert shape[0] == self.Wci.shape[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.shape[3], "Input Width Mismatched!"
        return (
            torch.zeros(batch_size, hidden, shape[0], shape[1], device=device),
            torch.zeros(batch_size, hidden, shape[0], shape[1], device=device),
        )


## Multi-layer ConvLSTM implementation
class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    """
    Convolutional LSTM model for processing spatio-temporal data.
    A ConvLSTM model is a sequence of ConvLSTM cells that processes the input tensor over multiple steps.
    If len(hidden_channels) == 1, it behaves like a standard ConvLSTM cell alone.

    Methods:
        __init__: Initializes the ConvLSTM model with input channels, hidden channels, kernel size, step, and effective step.
        reset_hidden_state: Resets the hidden states h and c of the ConvLSTM model to zeros.
        forward: Performs the forward pass of the ConvLSTM model.

    Attributes:
        input_channels: Number of channels for the first input feature map. Correspond to the inputs of ConvLSTMCell.
        hidden_channels: List of hidden channels for each layer. Correspond to the outputs of ConvLSTMCell.
        kernel_size: Size of the convolutional kernel in Convolutional layers of the ConvLSTM cell.
        num_layers: Number of ConvLSTM layers.
        step: Number of steps to process in the forward pass.
        effective_step: List of effective steps to record outputs.
        _all_layers: List of all ConvLSTM cells in the model. Each are callable objects that refers to the corresponding ConvLSTM cell.
        next_hidden_state: List of next hidden states for each layer. Will be use for the next window pass on the ConvLSTM.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: list[int],
        kernel_size: int | tuple,
        step: int = 1,
        effective_step: list[int] = [0, 1],
    ) -> None:
        super().__init__()
        assert (
            step >= 1
        ), "n_step must be at least 1 to find temporal dependencies and to register the next hidden state."

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step

        self._all_layers = nn.ModuleList()
        for i in range(self.num_layers):
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                kernel_size=self.kernel_size,
            )
            self._all_layers.append(cell)

        self.next_hidden_state = [None] * self.num_layers

    def reset_hidden_state(self):
        """
        Reset the hidden state of the ConvLSTM model.
        For instance, when we start a new epoch in the training, we want to reset the hidden state.
        """
        self.next_hidden_state = [None] * self.num_layers

    def forward(
        self,
        input: torch.Tensor,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor]]:
        """
        Forward pass of the ConvLSTM model.
        A ConvLSTM model is a sequence of ConvLSTM cells that processes the input tensor over multiple steps.
        For each step, the input tensor is passed through all layers.
        Each layer is a ConvLSTM cell that takes the output of the previous layer as input.

        Args:
            input (torch.Tensor): Input tensor of shape (self.step, input_channels, height, width)

        Returns:
            Tuple containing:
                - List of output tensors for each effective step: [h_0, h_1, ..., h_step]
                - Tuple of the last hidden state and cell state tensors (h_lastlayer_laststep, c_lastlayer_laststep)

        Nota bene:
            We initialize the internal state (hidden and cell states) for each layer in the first step.
            And that for each time window (of size self.step). Whether the model is in training or inference mode, the internal states are reset.
            Indeed, as the time window are overlapping, the internal state of the
        """
        internal_state = []
        outputs = []

        input = (
            input.unsqueeze(1) if input.dim() == 4 else input
        )  # Ensure input is 5D: (step, 1, channels, height, width)

        for step_idx in range(self.step):
            x = input[step_idx]

            for layer_idx, cell in enumerate(self._all_layers):
                if step_idx == 0:
                    if self.next_hidden_state[layer_idx] is None:
                        bsize, _, height, width = x.shape
                        h, c = cell.init_hidden(
                            batch_size=bsize,
                            hidden=self.hidden_channels[layer_idx],
                            shape=(height, width),
                            device=x.device,
                        )
                    else:
                        # if next_hidden_state is provided, use it
                        h, c = self.next_hidden_state[layer_idx]
                    internal_state.append((h, c))

                # do forward
                h, c = internal_state[layer_idx]
                new_h, new_c = cell(x, h, c)
                internal_state[layer_idx] = (new_h, new_c)
                # output of the current layer is the input for the next layer
                x = new_h

                ### if step_idx == 1 and not self.training: # <------- Uncomment this line to use persistant hidden states only during inference
                if step_idx == 1:
                    # register the hidden state for the next time window
                    h_save, c_save = new_h.clone().detach(), new_c.clone().detach()
                    self.next_hidden_state[layer_idx] = (h_save, c_save)

            # only record effective steps of the last layer
            if step_idx in self.effective_step:
                outputs.append(new_h)

        return outputs, (new_h, new_c)
