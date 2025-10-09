import torch
import pytest
import gc

from magnetics_diagnostic_analysis.project_vae.setting_vae import config
from magnetics_diagnostic_analysis.project_vae.model.lstm_vae import LengthAwareLSTMEncoder, LengthAwareLSTMDecoder, LSTMBetaVAE
from magnetics_diagnostic_analysis.ml_tools.metrics import vae_loss_function



def test_length_aware_lstm_encoder():
    # Parameters
    batch_size = 8
    input_dim = 4
    hidden_dim = 16
    latent_dim = 2
    num_layers = 2
    n_time = 50

    # Dynthetic data
    length_foo = torch.randint(low=10, high=n_time+1, size=(batch_size,)).to(config.DEVICE)
    seq_length = max(length_foo).item()
    x_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
    x_foo.requires_grad = True
    
    # Model
    encoder_foo = LengthAwareLSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers).to(config.DEVICE)
    mean_foo, logvar_foo, (h_final, c_final) = encoder_foo(x_foo, length_foo)
    
    # Tests
    assert mean_foo.shape == (batch_size, latent_dim), f"Expected mean shape: {(batch_size, latent_dim)}, got: {mean_foo.shape}"
    assert logvar_foo.shape == (batch_size, latent_dim), f"Expected logvar shape: {(batch_size, latent_dim)}, got: {logvar_foo.shape}"
    assert h_final.shape == (num_layers, batch_size, hidden_dim), f"Expected h_final shape: {(num_layers, batch_size, hidden_dim)}, got: {h_final.shape}"
    assert c_final.shape == (num_layers, batch_size, hidden_dim), f"Expected c_final shape: {(num_layers, batch_size, hidden_dim)}, got: {c_final.shape}"
    
    assert mean_foo.dtype == torch.float32, "Mean should be float32"
    assert logvar_foo.dtype == torch.float32, "Logvar should be float32"
    assert h_final.dtype == torch.float32, "Hidden state should be float32"
    assert c_final.dtype == torch.float32, "Cell state should be float32"
    
    assert not torch.isnan(mean_foo).any(), "Mean contains NaN values"
    assert not torch.isnan(logvar_foo).any(), "Logvar contains NaN values"
    assert not torch.isinf(mean_foo).any(), "Mean contains infinite values"
    assert not torch.isinf(logvar_foo).any(), "Logvar contains infinite values"
    
    for test_batch_size in [1, 5, 20]:
        length_test = torch.randint(low=10, high=n_time+1, size=(test_batch_size,)).to(config.DEVICE)
        seq_length_test = max(length_test).item()
        x_test = torch.randn(test_batch_size, seq_length_test, input_dim).to(config.DEVICE)
        mean_test, logvar_test, (h_test, c_test) = encoder_foo(x_test, length_test)
        assert mean_test.shape == (test_batch_size, latent_dim), f"Failed for batch size {test_batch_size}"
        assert logvar_test.shape == (test_batch_size, latent_dim), f"Failed for batch size {test_batch_size}"
    
    for min_length, max_length in [(10, 50), (100, 200), (1, 10)]:
        length_various = torch.randint(low=min_length, high=max_length+1, size=(batch_size,)).to(config.DEVICE)
        seq_length_various = max(length_various).item()
        x_various = torch.randn(batch_size, seq_length_various, input_dim).to(config.DEVICE)
        mean_various, logvar_various, _ = encoder_foo(x_various, length_various)
        assert mean_various.shape == (batch_size, latent_dim), f"Failed for length range {min_length}-{max_length}"
        assert logvar_various.shape == (batch_size, latent_dim), f"Failed for length range {min_length}-{max_length}"
    
    encoder_copy = LengthAwareLSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers).to(config.DEVICE)
    encoder_copy.load_state_dict(encoder_foo.state_dict())
    mean_copy, logvar_copy, _ = encoder_copy(x_foo, length_foo)
    assert torch.allclose(mean_foo, mean_copy, atol=1e-6), "Outputs not reproducible with same weights"
    assert torch.allclose(logvar_foo, logvar_copy, atol=1e-6), "Outputs not reproducible with same weights"
    
    #total_loss = mean_foo.sum() + logvar_foo.sum()
    #total_loss.backward()
    #for name, param in encoder_foo.named_parameters():
    #    assert param.grad is not None, f"Parameter {name} has no gradient"
    #    assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

    del encoder_foo
    del encoder_copy
    torch.cuda.empty_cache()
    gc.collect()

@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("latent_dim,hidden_dim,input_dim", [
    (2, 4, 8),
    (8, 16, 32), 
    (4, 8, 16)
])

def test_decoder_dimensions(batch_size, latent_dim, hidden_dim, input_dim):
    # Parameters, synthetic data & model
    num_layers = 2
    seq_length = 50
    
    length_foo = torch.randint(low=10, high=seq_length+1, size=(batch_size,)).to(config.DEVICE)
    z_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)

    decoder = LengthAwareLSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers).to(config.DEVICE)
    masked_output, (h_out, c_out) = decoder(z_foo, length_foo)
    
    # Tests
    max_length = torch.max(length_foo).item()
    assert masked_output.shape == (batch_size, max_length, input_dim)
    assert h_out.shape == (num_layers, batch_size, hidden_dim)
    assert c_out.shape == (num_layers, batch_size, hidden_dim)
    
    for i in range(batch_size):
        seq_len = length_foo[i].item()
        if seq_len < max_length:
            masked_part = masked_output[i, seq_len:, :]
            assert torch.all(masked_part == 0), f"Sequence {i} not properly masked"

    del decoder
    torch.cuda.empty_cache()
    gc.collect()

def test_decoder_masking():
    # Parameters, synthetic data & model
    batch_size = 3
    latent_dim = 2
    hidden_dim = 16
    input_dim = 4
    num_layers = 2
    
    specific_lengths = torch.tensor([2, 5, 3]).to(config.DEVICE)
    z_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)
    
    decoder = LengthAwareLSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers).to(config.DEVICE)
    masked_output, _ = decoder(z_foo, specific_lengths)
    
    # Tests
    max_length = torch.max(specific_lengths).item()
    assert masked_output.shape == (batch_size, max_length, input_dim)
    
    assert torch.all(masked_output[0, 2:, :] == 0)  # Sequence 0 masked after position 2
    assert torch.all(masked_output[1, 5:, :] == 0)  # Sequence 1 masked after position 5
    assert torch.all(masked_output[2, 3:, :] == 0)  # Sequence 2 masked after position 3

    assert not torch.all(masked_output[0, :2, :] == 0)
    assert not torch.all(masked_output[1, :5, :] == 0)
    assert not torch.all(masked_output[2, :3, :] == 0)

    del decoder
    torch.cuda.empty_cache()
    gc.collect()



@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("input_dim,hidden_dim,latent_dim", [
    (2, 4, 8),
    (8, 16, 32), 
    (4, 8, 16)
])
@pytest.mark.parametrize("bptt_steps", [20, 50, None])
def test_vae_dimensions(batch_size, input_dim, hidden_dim, latent_dim, bptt_steps):
    # Parameters, synthetic data & model
    num_layers = 2
    n_time = 50

    length_foo = torch.randint(low=10, high=n_time+1, size=(batch_size,)).to(config.DEVICE)
    seq_length = max(length_foo).item()
    x_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
 
    vae = LSTMBetaVAE(input_dim, hidden_dim, latent_dim, num_layers, bptt_steps=bptt_steps).to(config.DEVICE)
    output, mean, logvar = vae(x_foo, length_foo)
    
    # Tests
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert output.shape == (batch_size, torch.max(length_foo).item(), input_dim)
    assert output.shape == x_foo.shape
    
    for i in range(batch_size):
        seq_len = length_foo[i].item()
        if seq_len < seq_length:
            masked_part = output[i, seq_len:, :]
            assert torch.all(masked_part == 0), f"Sequence {i} not properly masked"

    del vae
    torch.cuda.empty_cache()
    gc.collect()





def test_vae_loss_function():
    # Parameters & synthetic data
    batch_size = 8
    input_dim = 4
    hidden_dim = 16
    latent_dim = 2
    num_layers = 2
    n_time = 50

    length_foo = torch.randint(low=10, high=n_time+1, size=(batch_size,)).to(config.DEVICE)
    seq_length = max(length_foo).item()
    x_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
    x_recon_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
    z_mean_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)
    z_logvar_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)

    beta = 2.0
    loss, loss_mse, loss_kld = vae_loss_function(x_foo, x_recon_foo, z_mean_foo, z_logvar_foo, length_foo, beta)

    assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
    assert loss_mse.shape == torch.Size([]), f"MSE loss should be scalar, got {loss_mse.shape}"
    assert loss_kld.shape == torch.Size([]), f"KLD loss should be scalar, got {loss_kld.shape}"

    assert loss.dim() == 0, "Total loss should be a scalar"
    assert loss_mse.dim() == 0, "MSE loss should be a scalar"
    assert loss_kld.dim() == 0, "KLD loss should be a scalar"

    assert loss.item() >= 0, f"Total loss should be non-negative, got {loss.item()}"
    assert loss_mse.item() >= 0, f"MSE loss should be non-negative, got {loss_mse.item()}"
    assert loss_kld.item() >= 0, f"KLD loss should be non-negative, got {loss_kld.item()}"

    expected_loss = loss_mse + beta * loss_kld
    assert torch.allclose(loss, expected_loss, atol=1e-6), f"Total loss should be MSE + beta*KLD. Got {loss.item()}, expected {expected_loss.item()}"

    del x_foo, x_recon_foo, z_mean_foo, z_logvar_foo
    gc.collect()

def test_vae_loss_beta_values():
    batch_size = 5
    seq_length = 50
    input_dim = 4
    latent_dim = 2

    torch.manual_seed(42)
    x_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
    x_recon_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
    z_mean_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)
    z_logvar_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)
    length_foo = torch.randint(low=10, high=seq_length+1, size=(batch_size,)).to(config.DEVICE)

    beta_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    for beta in beta_values:
        loss, loss_mse, loss_kld = vae_loss_function(x_foo, x_recon_foo, z_mean_foo, z_logvar_foo, length_foo, beta)
        assert loss.dim() == 0, f"Loss should be scalar for beta={beta}"
        assert loss_mse.dim() == 0, f"MSE loss should be scalar for beta={beta}"
        assert loss_kld.dim() == 0, f"KLD loss should be scalar for beta={beta}"
        if beta == 0.0:
            assert torch.allclose(loss, loss_mse, atol=1e-6), f"For beta=0, loss should equal MSE. Got {loss.item()} vs {loss_mse.item()}"
        
    del x_foo, x_recon_foo, z_mean_foo, z_logvar_foo
    gc.collect()
    
def test_vae_loss_different_batch_sizes():
    batch_sizes = [1, 3, 10]
    input_dim = 4
    latent_dim = 2
    seq_length = 50

    for batch_size in batch_sizes:
        length_foo = torch.randint(low=10, high=seq_length+1, size=(batch_size,)).to(config.DEVICE)
        x_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
        x_recon_foo = torch.randn(batch_size, seq_length, input_dim).to(config.DEVICE)
        z_mean_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)
        z_logvar_foo = torch.randn(batch_size, latent_dim).to(config.DEVICE)

        loss, loss_mse, loss_kld = vae_loss_function(
            x_foo, x_recon_foo, z_mean_foo, z_logvar_foo, length_foo, beta=1.0
        )
        assert loss.dim() == 0, f"Loss should be scalar for batch_size={batch_size}"

    del x_foo, x_recon_foo, z_mean_foo, z_logvar_foo
    gc.collect()



if __name__ == "__main__":
    config.update(DEVICE="cpu")

    print("Running LengthAwareLSTMEncoder tests...")
    test_length_aware_lstm_encoder()
    print("All LengthAwareLSTMEncoder tests passed successfully! \n")


    print("Running LengthAwareLSTMDecoder tests...")
    test_decoder_dimensions()
    test_decoder_masking()
    print("All LengthAwareLSTMDecoder tests passed successfully!\n")


    print("Running LSTMBetaVAE tests...")
    test_vae_dimensions()
    print("All LSTMBetaVAE tests passed successfully!\n")

    print("Running VAE loss function tests...")
    test_vae_loss_function()
    test_vae_loss_beta_values()
    test_vae_loss_different_batch_sizes()
    print("All VAE loss function tests passed successfully!\n")

    print("\nAll tests passed successfully!")

    from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import select_torch_device
    config.update(DEVICE=select_torch_device())
