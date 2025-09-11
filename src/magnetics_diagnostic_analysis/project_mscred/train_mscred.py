import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.ml_tools.metrics import mscred_loss_function
from magnetics_diagnostic_analysis.ml_tools.train_callbacks import EarlyStopping, LRScheduling, GradientClipping
from magnetics_diagnostic_analysis.project_mscred.utils.dataloader_building import create_data_loaders
from magnetics_diagnostic_analysis.project_mscred.model.mscred import MSCRED


def train(model: nn.Module, dataLoader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device, valid_loader: torch.utils.data.DataLoader = None):
    torch.cuda.empty_cache()
    model = model.to(device)
    print("------training on {}-------".format(device))

    early_stopper = EarlyStopping(min_delta=0.01, patience=10)
    lr_scheduler = LRScheduling(optimizer, mode='min', factor=0.66, patience=3, min_lr=1e-6, min_delta=0.001)
    gradient_clipper = GradientClipping(max_norm=2.0)
    history = {'train_loss': [], 'valid_loss': []}

    for epoch in range(epochs):
        model.reset_lstm_hidden_states()

        # Training
        model.train()
        train_loss_sum, n = 0.0, 0
        for x in tqdm(dataLoader):
            x = x.to(device)
            optimizer.zero_grad()

            x_recon = model(x)
            loss = mscred_loss_function(x_recon, x)
            loss.backward()
            gradient_clipper.on_backward_end(model)
            optimizer.step()

            train_loss_sum += loss.item()
            n += 1
            
        train_loss = train_loss_sum / n
        history['train_loss'].append(train_loss)


        # Validation (if valid_loader is not None)
        valid_loss = None
        if valid_loader is not None:
            model.eval()
            valid_loss_sum, valid_n = 0.0, 0
            with torch.no_grad():
                for x_valid in valid_loader:
                    x_valid = x_valid.to(device)
                    x_recon_valid = model(x_valid)
                    loss_valid = mscred_loss_function(x_recon_valid, x_valid)
                    valid_loss_sum += loss_valid.item()
                    valid_n += 1
            valid_loss = valid_loss_sum / valid_n
            history['valid_loss'].append(valid_loss)

        current_loss = valid_loss if valid_loader is not None else train_loss

        print("[Epoch %d/%d] [Train loss: %f] %s" % (
            epoch+1, epochs, train_loss, 
            f"[Val loss: {valid_loss:.4f}]" if valid_loss is not None else "No valid"
        ))
        
        if early_stopper.check_stop(current_loss, model):
            print(f"Early stopping at epoch {epoch + 1} with loss {current_loss:.4f}")
            print(f"Restoring best weights for model.")
            early_stopper.restore_best_weights(model)
            break

        lr_scheduler.step(current_loss)

        path = Path(__file__).absolute().parent / "checkpoints/model_checkpointed.pth"
        torch.save(model.state_dict(), path)

    return history, model

def plot_history(history_train: list, history_valid: list) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_train, 'b-', linewidth=2, label='Train Loss')
    plt.plot(history_valid, 'r-', linewidth=2, label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = Path(__file__).absolute().parent.parent.parent.parent / "results/figures/mscred/last_training_history.png"
    plt.savefig(path)
    return None


def main():
    device = config.DEVICE
    # data_foo = np.random.randn(2000, *config.DATA_SHAPE)  # Same shape as the window_matrix

    path = config.DIR_PREPROCESSED_DATA / "signature_matrices.npy"
    data = np.load(path)

    train_loader, valid_loader, test_loader = create_data_loaders(
        data=data,
        batch_size=config.BATCH_SIZE,
        set_separations=config.SET_SEPARATIONS,
        gap_time=config.GAP_TIME,
        device=device
    )

    mscred = MSCRED(
        encoder_in_channel=config.DATA_SHAPE[0],
        deep_channel_sizes=config.DEEP_CHANNEL_SIZES,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_timesteps=config.LSTM_TIMESTEPS,
        lstm_effective_timesteps=config.LSTM_EFFECTIVE_TIMESTEPS
    )

    optimizer = torch.optim.Adam(mscred.parameters(), lr = config.FIRST_LEARNING_RATE)
    model_name_to_continue = "model0"
    model_name_register = config.BEST_MODEL_NAME

    continue_training = False
    if continue_training:
        mscred.load_state_dict(torch.load(config.DIR_MODEL_PARAMS / f"{model_name_to_continue}.pth"))
    
    # Train
    try:
        history, trained_mscred = train(
            mscred, 
            train_loader, 
            optimizer, 
            epochs=config.N_EPOCHS, 
            device=config.DEVICE, 
            valid_loader=valid_loader
        )
        torch.save(trained_mscred.state_dict(), config.DIR_MODEL_PARAMS / f"{model_name_register}.pth")
        #config.update({"BEST_MODEL_NAME": model_name_register})

        plot_history(history['train_loss'], history['valid_loss'])


    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        if 'trained_mscred' in locals():
            torch.save(trained_mscred.state_dict(), config.DIR_MODEL_PARAMS / f"{model_name_register}_interrupted.pth")
        else:
            torch.save(mscred.state_dict(), config.DIR_MODEL_PARAMS / f"{model_name_register}_interrupted.pth")
        print(f"Model saved to: {config.DIR_MODEL_PARAMS / f'{model_name_register}_interrupted.pth'}")
        return

    torch.save(trained_mscred.state_dict(), config.DIR_MODEL_PARAMS / f"{model_name_register}.pth")
    #config.update({"BEST_MODEL_NAME": model_name_register})

    plot_history(history['train_loss'], history['valid_loss'])



if __name__ == "__main__":
    torch.backends.cudnn.enabled = False        # problem to fix in the future
    main()