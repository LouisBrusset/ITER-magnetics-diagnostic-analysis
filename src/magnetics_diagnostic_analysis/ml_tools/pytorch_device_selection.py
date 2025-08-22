import torch

def print_torch_info():
    print("\nTorch version? ", torch.__version__)
    print("Cuda?          ", torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\nGPU number : {device_count}")

        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nNo NVIDIA GPU is available for PyTorch.")


def select_torch_device() -> torch.device:
    # We want to force the use of one specific GPU if available and only one.
    # Indeed, the batchs must be treated in chronologic order, due to the ConvLSTM module and the time deque between batches.

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("\n\nChoosen device =", device)
    return device


if __name__ == "__main__":

    print_torch_info()
    
    device = select_torch_device()

    print(f"Using device: {device}")

