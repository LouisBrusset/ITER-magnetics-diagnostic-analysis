import torch
import pytest
import sys
import os

# Add path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import classes (adjust these imports according to your structure)
from magnetics_diagnostic_analysis.project_mscred.model.mscred import attention, CnnEncoder, CnnDecoder, Conv_LSTM, MSCRED

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tests for attention function
def test_attention():
    """Test attention function with different scale factors"""
    n_effective_timestep = 5
    channels = 64
    height = 32
    width = 32

    conv_lstm_out = torch.randn(n_effective_timestep, channels, height, width)
    
    # Basic test
    attention_output = attention(conv_lstm_out)
    
    # Shape verification
    assert attention_output.shape == (channels, height, width), \
        f"Expected shape: {(channels, height, width)}, obtained: {attention_output.shape}"
    
    # Value verification
    assert not torch.isnan(attention_output).any(), "Result contains NaN values"
    assert not torch.isinf(attention_output).any(), "Result contains infinite values"
    
    # Tests with different scale factors
    for rescale in [0.1, 1.0, 5.0, 10.0, 100.0]:
        output = attention(conv_lstm_out, rescale_factor=rescale)
        assert output.shape == (channels, height, width), \
            f"Error with rescale_factor={rescale}: wrong shape"

# Tests for CnnEncoder
def test_cnn_encoder():
    """Test CnnEncoder with different input sizes"""
    batch_size = 10
    height = 32
    width = 32
    in_channels = 3
    deep_channels = [32, 64, 128]

    X = torch.randn(batch_size, in_channels, height, width)
    encoder = CnnEncoder(in_channels_encoder=in_channels, deep_channel_sizes=deep_channels)
    conv1_out, conv2_out, conv3_out = encoder(X)

    # Shape verification
    assert conv1_out.shape == (batch_size, 32, height, width), \
        f"conv1_out shape expected: {(batch_size, 32, height, width)}, obtained: {conv1_out.shape}"
    
    assert conv2_out.shape == (batch_size, 64, height//2, width//2), \
        f"conv2_out shape expected: {(batch_size, 64, height//2, width//2)}, obtained: {conv2_out.shape}"
    
    assert conv3_out.shape == (batch_size, 128, height//4, width//4), \
        f"conv3_out shape expected: {(batch_size, 128, height//4, width//4)}, obtained: {conv3_out.shape}"

    # Value verification
    for name, tensor in [('conv1_out', conv1_out), ('conv2_out', conv2_out), ('conv3_out', conv3_out)]: 
        assert not torch.isnan(tensor).any(), f"{name} contains NaN" 
        assert not torch.isinf(tensor).any(), f"{name} contains infinite values"

    # Tests with different input sizes
    test_sizes = [(32, 32), (64, 64), (128, 128)]
    for h, w in test_sizes: 
        X_test = torch.randn(batch_size, in_channels, h, w) 
        c1, c2, c3 = encoder(X_test) 
        assert c1.shape == (batch_size, 32, h, w) 
        assert c2.shape == (batch_size, 64, h//2, w//2) 
        assert c3.shape == (batch_size, 128, h//4, w//4)

# Tests for CnnDecoder
def test_cnn_decoder():
    """Test CnnDecoder with different input sizes"""
    batch_size = 10
    base_height = 4
    base_width = 4
    encoder_in_channels = 3
    deep_channels = [32, 64, 128]

    # Simulated outputs from ConvLSTM layers
    conv1_lstm_out = torch.randn(batch_size, 32, base_height*8, base_width*8)
    conv2_lstm_out = torch.randn(batch_size, 64, base_height*4, base_width*4)
    conv3_lstm_out = torch.randn(batch_size, 128, base_height*2, base_width*2)

    decoder = CnnDecoder(in_channels_encoder=encoder_in_channels, deep_channel_sizes=deep_channels)
    output = decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out)

    # Verification
    expected_shape = (batch_size, encoder_in_channels, base_height*8, base_width*8)
    assert output.shape == expected_shape, f"Expected shape: {expected_shape}, obtained: {output.shape}"
    assert not torch.isnan(output).any(), "Result contains NaNs"
    assert not torch.isinf(output).any(), "Result contains infinite values"

    # Tests with different sizes
    test_sizes = [(4, 4), (8, 8), (16, 16)]
    for h, w in test_sizes:
        conv1 = torch.randn(batch_size, 32, h*8, w*8)
        conv2 = torch.randn(batch_size, 64, h*4, w*4)
        conv3 = torch.randn(batch_size, 128, h*2, w*2)

        out = decoder(conv1, conv2, conv3)
        assert out.shape == (batch_size, 3, h*8, w*8), f"Failed for base size {(h,w)}: got {out.shape}"

# Tests for Conv_LSTM
def test_conv_lstm():
    """Test Conv_LSTM with different input sizes"""
    batch_size = 5
    height = 16
    width = 16
    layers = 1
    deep_channels = [16, 32, 64]
    timesteps = 5
    attention_timesteps = [1, 2, 3, 4]

    # Simulated outputs from encoder
    conv1_out = torch.randn(batch_size, 16, height, width).to(device)
    conv2_out = torch.randn(batch_size, 32, height//2, width//2).to(device)
    conv3_out = torch.randn(batch_size, 64, height//4, width//4).to(device)

    conv_lstm = Conv_LSTM(
        deep_channel_sizes=deep_channels, 
        num_layers=layers, 
        n_timesteps=timesteps, 
        n_effective_timesteps=attention_timesteps
    ).to(device)
    
    conv1_lstm, conv2_lstm, conv3_lstm = conv_lstm(conv1_out, conv2_out, conv3_out)

    # Shape verification
    assert conv1_lstm.shape == (batch_size, 16, height, width), \
        f"conv1_lstm shape expected: {(batch_size, 16, height, width)}, obtained: {conv1_lstm.shape}"
    
    assert conv2_lstm.shape == (batch_size, 32, height//2, width//2), \
        f"conv2_lstm shape expected: {(batch_size, 32, height//2, width//2)}, obtained: {conv2_lstm.shape}"
    
    assert conv3_lstm.shape == (batch_size, 64, height//4, width//4), \
        f"conv3_lstm expected shape: {(batch_size, 64, height//4, width//4)}, got: {conv3_lstm.shape}"

    # Value verification
    for name, tensor in [('conv1_lstm', conv1_lstm), ('conv2_lstm', conv2_lstm), ('conv3_lstm', conv3_lstm)]:
        assert not torch.isnan(tensor).any(), f"{name} contains NaNs"
        assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
        assert tensor.requires_grad, f"{name} does not preserve the gradient"

    # Tests with different image sizes
    test_sizes = [(16, 16), (8, 8), (32, 32)]
    for h, w in test_sizes:
        # Recreate inputs with new dimensions
        conv1 = torch.randn(batch_size, 16, h, w).to(device)
        conv2 = torch.randn(batch_size, 32, h//2, w//2).to(device)
        conv3 = torch.randn(batch_size, 64, h//4, w//4).to(device)
        
        # Create and test ConvLSTM
        conv_lstm = Conv_LSTM(
            deep_channel_sizes=deep_channels, 
            num_layers=layers, 
            n_timesteps=timesteps, 
            n_effective_timesteps=attention_timesteps
        ).to(device)
        
        out1, out2, out3 = conv_lstm(conv1, conv2, conv3) 
        
        # Verification
        assert out1.shape == (batch_size, 16, h, w)
        assert out2.shape == (batch_size, 32, h//2, w//2) 
        assert out3.shape == (batch_size, 64, h//4, w//4)

# Tests for MSCRED
def test_mscred():
    """Test complete MSCRED model with different input sizes"""
    batch_size = 5
    in_channels = 2
    height = 8
    width = 8
    layers = 1
    deep_channels = [16, 32, 64]
    timesteps = 5
    effective_steps = [1, 2, 4]

    X = torch.randn(batch_size, in_channels, height, width).to(device)

    mscred = MSCRED(
        encoder_in_channel=in_channels, 
        deep_channel_sizes=deep_channels, 
        lstm_num_layers=layers, 
        lstm_timesteps=timesteps, 
        lstm_effective_timesteps=effective_steps
    ).to(device)
    
    X_recon = mscred(X).cpu()

    # Shape verification
    assert X.shape == X_recon.shape, f"X shape expected: {X.shape}, obtained: {X_recon.shape}"

    # Value verification
    assert not torch.isnan(X_recon).any(), "X_recon contains NaNs"
    assert not torch.isinf(X_recon).any(), "X_recon contains infinite values"
    assert X_recon.requires_grad, "X_recon does not preserve the gradient"

    # Tests with different image sizes
    test_sizes = [(32, 32), (36, 36), (28, 28)]
    for h, w in test_sizes:
        # Recreate inputs with new dimensions
        X_test = torch.randn(batch_size, in_channels, h, w).to(device)
        
        # Create and test the model
        mscred = MSCRED(
            encoder_in_channel=in_channels, 
            deep_channel_sizes=deep_channels, 
            lstm_num_layers=layers, 
            lstm_timesteps=timesteps, 
            lstm_effective_timesteps=effective_steps
        ).to(device)
        
        X_recon = mscred(X_test).cpu()
        
        # Verification
        assert X_recon.shape == (batch_size, in_channels, h, w)



# Test for gradient propagation
def test_gradient_flow():
    """Test gradient propagation through all model components"""
    batch_size = 5
    in_channels = 2
    height = 8
    width = 8
    layers = 2
    deep_channels = [16, 32, 64]
    timesteps = 5
    effective_steps = [1, 2, 4]

    # Create model and sample batch
    model = MSCRED(
        encoder_in_channel=in_channels, 
        deep_channel_sizes=deep_channels, 
        lstm_num_layers=layers, 
        lstm_timesteps=timesteps, 
        lstm_effective_timesteps=effective_steps
    ).to(device)
    
    sample_batch = torch.randn(batch_size, in_channels, height, width).to(device)
    
    # Test gradient flow
    model.train()
    output = model(sample_batch)
    loss = torch.mean(output)  # Simple loss for testing
    loss.backward()
    
    # Check if encoder parameters receive gradients
    encoder_has_gradients = False
    for name, param in model.named_parameters():
        if 'cnn_encoder' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                encoder_has_gradients = True
                print(f"Encoder gradient flow: {name} - {grad_norm}")
    
    assert encoder_has_gradients, "Encoder parameters are not receiving gradients"
    
    # Check if ConvLSTM parameters receive gradients  
    conv_lstm_has_gradients = False
    for name, param in model.named_parameters():
        if 'conv_lstm' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                conv_lstm_has_gradients = True
                print(f"ConvLSTM gradient flow: {name} - {grad_norm}")

    assert conv_lstm_has_gradients, "ConvLSTM parameters are not receiving gradients"

    # Check if decoder parameters receive gradients
    decoder_has_gradients = False
    for name, param in model.named_parameters():
        if 'cnn_decoder' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                decoder_has_gradients = True
                print(f"Decoder gradient flow: {name} - {grad_norm}")

    assert decoder_has_gradients, "Decoder parameters are not receiving gradients"
    
    # Check if attention parameters receive gradients
    attention_has_gradients = False
    for name, param in model.named_parameters():
        if 'attention' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                attention_has_gradients = True
                print(f"Attention gradient flow: {name} - {grad_norm}")

    assert attention_has_gradients, "Attention parameters are not receiving gradients"

if __name__ == "__main__":
    # Run tests
    test_attention()
    print("Attention tests passed!")
    
    test_cnn_encoder()
    print("CNN Encoder tests passed!")
    
    test_cnn_decoder()
    print("CNN Decoder tests passed!")
    
    test_conv_lstm()
    print("Conv LSTM tests passed!")
    
    test_mscred()
    print("MSCRED tests passed!")
    
    test_gradient_flow()
    print("Gradient flow tests passed!")
    
    print("\nAll tests passed successfully!")