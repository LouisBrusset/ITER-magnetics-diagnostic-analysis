import numpy as np
import torch





class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric has stopped improving.

    Methods:
    check_stop(current_loss: float, model) -> bool:
        Check if training should be stopped based on the current loss.
    
    restore_best_weights(model) -> None:
        Restore the model weights from the best epoch.
    """
    def __init__(self, min_delta: float = 0.001, patience: int = 5) -> None:
        """
        Early stopping to terminate training when a monitored metric has stopped improving.

        Parameters:
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.best_state_dict = None  # <- Ajout

    def check_stop(self, current_loss: float, model) -> bool:
        """
        Check if training should be stopped based on the current loss.

        Parameters:
        current_loss: The loss value for the current epoch.
        model: The model whose state_dict will be saved if it is the best so far.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()  # <- Save initial weights
            return False

        relative_change = abs((self.best_loss - current_loss) / (self.best_loss + 1e-8))

        if relative_change < self.min_delta:
            self.counter += 1
        else:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state_dict = model.state_dict()  # <- Save new best weights

        return self.counter >= self.patience

    def restore_best_weights(self, model) -> None:
        """
        Restore the model weights from the best epoch.

        Parameters:
        model: The model whose weights will be restored.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)