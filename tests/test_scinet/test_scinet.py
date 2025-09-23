import torch
import pytest
import gc

from magnetics_diagnostic_analysis.project_vae.setting_vae import config
from magnetics_diagnostic_analysis.project_scinet.model.scinet import PendulumNet, SciNetEncoder, QuestionDecoder
from magnetics_diagnostic_analysis.ml_tools.metrics import scinet_loss


def test_scinet_encoder():

    input_size = 50
    latent_size = 3
    hidden_sizes = [500, 100]
    batch_size = 10
    
    encoder = SciNetEncoder(
        input_size=input_size,
        latent_size=latent_size,
        hidden_sizes=hidden_sizes
    )
    
    x = torch.randn((batch_size, input_size))
    mean, logvar = encoder(x)
    
    assert mean.shape == (batch_size, latent_size)
    assert logvar.shape == (batch_size, latent_size)
    assert not torch.isnan(mean).any()
    assert not torch.isnan(logvar).any()
    
    loss = mean.sum() + logvar.sum()
    loss.backward()
    
    for param in encoder.parameters():
        assert param.grad is not None
    
    del encoder, x, mean, logvar, loss
    gc.collect()


def test_question_decoder():

    latent_size = 3
    question_size = 1
    output_size = 1
    hidden_sizes = [64, 32]
    batch_size = 10
    
    decoder = QuestionDecoder(
        latent_size=latent_size,
        question_size=question_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes
    )
    
    z = torch.randn((batch_size, latent_size))
    question = torch.randn((batch_size, question_size))
    output = decoder(z, question)
    
    assert output.shape == (batch_size, output_size)
    assert not torch.isnan(output).any()
    
    loss = output.sum()
    loss.backward()
    
    for param in decoder.parameters():
        assert param.grad is not None
    
    del decoder, z, question, output, loss
    gc.collect()



def test_pendulum_net():

    input_size = 50
    enc_hidden_sizes = [128, 64]
    latent_size = 3
    question_size = 1
    dec_hidden_sizes = [64, 32]
    output_size = 1
    batch_size = 10

    model = PendulumNet(
        input_size=input_size,
        enc_hidden_sizes=enc_hidden_sizes,
        latent_size=latent_size,
        question_size=question_size,
        dec_hidden_sizes=dec_hidden_sizes,
        output_size=output_size
    )

    x = torch.randn((batch_size, input_size))
    question = torch.randn((batch_size, question_size))
    possible_answer, mean, logvar = model(x, question)

    assert possible_answer.shape == (batch_size, output_size)
    assert mean.shape == (batch_size, latent_size)
    assert logvar.shape == (batch_size, latent_size)
    assert not torch.isnan(possible_answer).any()
    assert not torch.isnan(mean).any()
    assert not torch.isnan(logvar).any()

    loss = possible_answer.sum() + mean.sum() + logvar.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
    
    del model, x, question, possible_answer, mean, logvar, loss
    gc.collect()



def test_scinet_loss():

    batch_size = 10
    output_size = 1
    latent_size = 3
    
    possible_answer = torch.randn((batch_size, output_size), requires_grad=True)
    a_corr = torch.randn((batch_size, output_size))
    mean = torch.randn((batch_size, latent_size), requires_grad=True)
    logvar = torch.randn((batch_size, latent_size), requires_grad=True)
    beta = 0.01

    total_loss, kld_loss, recon_loss = scinet_loss(possible_answer, a_corr, mean, logvar, beta)

    assert total_loss.shape == ()  # scalar
    assert kld_loss.shape == ()   # scalar
    assert recon_loss.shape == () # scalar
    assert total_loss.item() >= 0  # loss must be positive
    assert kld_loss.item() >= 0    # KLD must be positive
    assert recon_loss.item() >= 0  # reconstruction loss must be positive

    total_loss.backward()
    assert possible_answer.grad is not None
    assert mean.grad is not None
    assert logvar.grad is not None
    
    del possible_answer, a_corr, mean, logvar, total_loss, kld_loss, recon_loss
    gc.collect()




if __name__ == "__main__":
    config.update(DEVICE="cpu")

    print("\n--- Running SciNet Unit Tests ---\n")

    print("Testing SciNetEncoder...")
    test_scinet_encoder()
    print("SciNetEncoder tests passed.\n")

    print("Testing SciNetQuestionDecoder...")
    test_question_decoder()
    print("SciNetQuestionDecoder tests passed.\n")

    print("Testing PendulumNet...")
    test_pendulum_net()
    print("PendulumNet tests passed.\n")

    print("Testing SciNet Loss Function...")
    test_scinet_loss()
    print("SciNet Loss Function tests passed.\n\n")

    print("--- All tests passed successfully! ---\n")





    from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import select_torch_device
    config.update(DEVICE=select_torch_device())