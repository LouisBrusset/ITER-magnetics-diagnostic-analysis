import torch
import pytest
from magnetics_diagnostic_analysis.project_mscred.model.convlstm import ConvLSTMCell, ConvLSTM

# Définir le device (CPU ou GPU si disponible)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestConvLSTMCell:
    """Tests for ConvLSTMCell class"""
    
    def test_convlstm_cell_shapes(self):
        """Test the input/output shapes of ConvLSTMCell"""
        batch_size = 1
        input_channels = 3
        height, width = 32, 32
        kernel_size = 3
        hidden_channels = 64
        
        # Create input tensor
        x = torch.randn(batch_size, input_channels, height, width).to(device)
        
        # Initialize ConvLSTMCell
        convlstm_cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        ).to(device)
        
        # Initialize hidden states
        h, c = convlstm_cell.init_hidden(batch_size, hidden_channels, (height, width), device)
        
        # Forward pass
        new_h, new_c = convlstm_cell(x, h, c)
        
        # Assert shapes
        assert x.shape == (batch_size, input_channels, height, width)
        assert h.shape == (batch_size, hidden_channels, height, width)
        assert c.shape == (batch_size, hidden_channels, height, width)
        assert new_h.shape == (batch_size, hidden_channels, height, width)
        assert new_c.shape == (batch_size, hidden_channels, height, width)
        
        # Specific assertions from your test
        assert new_h.shape == (batch_size, hidden_channels, height, width)
        assert new_c.shape == new_h.shape

class TestConvLSTM:
    """Tests for ConvLSTM class"""
    
    def test_convlstm_shapes(self):
        """Test the input/output shapes of ConvLSTM"""
        batch_size = 1
        input_channels = 3
        hidden_channels = [32, 64, 128]
        height, width = 32, 32
        kernel_size = 3
        steps = 5
        effective_step = [0, 1, 2, 3, 4]
        
        # Create input tensor (sequence)
        x = torch.randn(steps, batch_size, input_channels, height, width).to(device)
        
        # Initialize ConvLSTM
        convlstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            step=steps,
            effective_step=effective_step
        ).to(device)
        
        # Forward pass
        outputs, (final_h, final_c) = convlstm(x)
        
        # Assert input shape
        assert x.shape == (steps, batch_size, input_channels, height, width)
        
        # Assert model properties
        assert len(convlstm.hidden_channels) == len(hidden_channels)
        assert convlstm.num_layers == len(hidden_channels)
        assert len(outputs) == len(effective_step)
        assert convlstm.effective_step == effective_step
        
        # Assert output shapes
        for i, step in enumerate(convlstm.effective_step):
            assert outputs[i].shape == (batch_size, hidden_channels[-1], height, width)
        
        # Assert final states shapes
        assert final_h.shape == (batch_size, hidden_channels[-1], height, width)
        assert final_c.shape == (batch_size, hidden_channels[-1], height, width)
        assert final_h.shape == final_c.shape
    
    def test_convlstm_different_effective_steps(self):
        """Test ConvLSTM with different effective step configurations"""
        batch_size = 2
        input_channels = 1
        hidden_channels = [16, 32]
        height, width = 16, 16
        kernel_size = 3
        steps = 4
        
        # Test with different effective step configurations
        test_cases = [
            [0, 1, 2, 3],  # All steps
            [0, 3],        # First and last
            [2],           # Only middle
            [1, 2]         # Middle two
        ]
        
        x = torch.randn(steps, batch_size, input_channels, height, width).to(device)
        
        for effective_step in test_cases:
            convlstm = ConvLSTM(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                step=steps,
                effective_step=effective_step
            ).to(device)
            
            outputs, (final_h, final_c) = convlstm(x)
            
            # Assert number of outputs matches effective steps
            assert len(outputs) == len(effective_step)
            
            # Assert each output has correct shape
            for output in outputs:
                assert output.shape == (batch_size, hidden_channels[-1], height, width)

# Tests supplémentaires pour edge cases
def test_convlstm_cell_gradients():
    """Test that gradients flow through ConvLSTMCell"""
    batch_size = 2
    input_channels = 4
    height, width = 16, 16
    hidden_channels = 32
    kernel_size = 3
    
    x = torch.randn(batch_size, input_channels, height, width).to(device)
    x.requires_grad = True
    
    convlstm_cell = ConvLSTMCell(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size
    ).to(device)
    
    h, c = convlstm_cell.init_hidden(batch_size, hidden_channels, (height, width), device)
    
    # Forward pass
    new_h, new_c = convlstm_cell(x, h, c)
    
    # Backward pass
    loss = new_h.sum() + new_c.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

if __name__ == "__main__":
    # Pour exécuter les tests directement sans pytest
    test_cell = TestConvLSTMCell()
    test_cell.test_convlstm_cell_shapes()
    
    test_lstm = TestConvLSTM()
    test_lstm.test_convlstm_shapes()
    test_lstm.test_convlstm_different_effective_steps()
    
    test_convlstm_cell_gradients()
    
    print("All tests passed!")