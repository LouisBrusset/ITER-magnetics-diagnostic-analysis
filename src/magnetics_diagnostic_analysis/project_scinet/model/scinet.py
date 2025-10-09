import torch
import torch.nn as nn


class SciNetEncoder(nn.Module):
    def __init__(self, 
                 input_size: int = 50, 
                 latent_size: int = 3, 
                 hidden_sizes: list[int] = [128, 64]
                 ) -> None:
        super().__init__()
        self.input_sizes = [input_size] + hidden_sizes[:-1]
        self.output_sizes = hidden_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(self.input_sizes, self.output_sizes)]
        )
        self.activations = nn.ModuleList(
            [nn.ELU() for _ in range(len(hidden_sizes))]
        )
        self.mean_layer = nn.Linear(hidden_sizes[-1], latent_size)
        self.logvar_layer = nn.Linear(hidden_sizes[-1], latent_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Forward pass through the encoder network. We loop over the layers and activations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            mean (torch.Tensor): Mean of the latent distribution of shape (batch_size, latent_size).
            logvar (torch.Tensor): Log-variance of the latent distribution of shape (batch_size, latent_size).
        """
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    

class QuestionDecoder(nn.Module):
    def __init__(self, 
                 latent_size: int = 3, 
                 question_size: int = 1, 
                 output_size: int = 1, 
                 hidden_sizes: list = [64, 32]
                 ) -> None:
        super().__init__()
        self.input_sizes = [latent_size + question_size] + hidden_sizes
        self.output_sizes = hidden_sizes + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(self.input_sizes, self.output_sizes)]
        )
        self.activations = nn.ModuleList(
            [nn.ELU() for _ in range(len(hidden_sizes))] + [nn.Identity()]
        )

    def forward(self, z: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder network. We concatenate the latent representation and the question,
        then loop over the layers and activations.

        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_size).
            question (torch.Tensor): Question tensor of shape (batch_size, question_size).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, output_size).
        """
        z = torch.cat([z, question], dim=-1)
        for layer, activation in zip(self.layers, self.activations):
            z = activation(layer(z))
        return z


class PendulumNet(nn.Module):
    def __init__(self, 
                 input_size: int = 50, 
                 enc_hidden_sizes: list[int] = [500, 100], 
                 latent_size: int = 10, 
                 question_size: int = 1,
                 dec_hidden_sizes: list[int] = [100, 100], 
                 output_size: int = 1
                 ) -> None:
        super().__init__()
        self.encoder = SciNetEncoder(input_size=input_size, latent_size=latent_size, hidden_sizes=enc_hidden_sizes)
        self.decoder = QuestionDecoder(latent_size=latent_size, question_size=question_size, output_size=output_size, hidden_sizes=dec_hidden_sizes)


    def forward(self, x, question):
        """
        Forward pass through the entire SciNet model: encoder, reparameterization, and decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            question (torch.Tensor): Question tensor of shape (batch_size, question_size).

        Returns:
            possible_answer (torch.Tensor): Output tensor of shape (batch_size, output_size).
            mean (torch.Tensor): Mean of the latent distribution of shape (batch_size, latent_size
            logvar (torch.Tensor): Log-variance of the latent distribution of shape (batch_size, latent_size).
        """
        mean, logvar = self.encoder(x)
        z = self.reparametrize(mean, logvar)
        possible_answer = self.decoder(z, question)
        return possible_answer, mean, logvar

    def reparametrize(self, mean, logvar):
        """
        Reparameterization trick to sample from N(mean, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std





