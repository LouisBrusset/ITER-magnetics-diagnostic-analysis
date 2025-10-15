import torch

from magnetics_diagnostic_analysis.project_vae.setting_vae import config
from magnetics_diagnostic_analysis.project_vae.model.lstm_vae import LSTMBetaVAE


def print_model_parameters(model, model_name="LSTMBetaVAE"):
    print(f"Model: {model_name}")
    print("=" * 60)

    total_params = 0
    for name, module in model.named_children():
        if hasattr(module, "parameters"):
            module_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            print(f"{name}: {module_params:,} parameters")
            total_params += module_params

            print("-" * 40)
            for sub_name, sub_module in module.named_children():
                if hasattr(sub_module, "parameters"):
                    sub_params = sum(
                        p.numel() for p in sub_module.parameters() if p.requires_grad
                    )
                    print(f"  ├─ {sub_name}: {sub_params:,} parameters")

                    if hasattr(sub_module, "named_parameters"):
                        print("  │   └─ Components:")
                        for param_name, param in sub_module.named_parameters():
                            if param.requires_grad:
                                print(
                                    f"  │      ├─ {param_name}: {param.numel():,} parameters "
                                    f"(shape: {tuple(param.shape)})"
                                )
            print("-" * 40)
            print()

    print("=" * 60)
    print(f"Total Trainable Parameters: {total_params:,}")

    print("\nArchitecture Details:")
    print("-" * 40)

    if hasattr(model, "encoder"):
        encoder = model.encoder
        print("Encoder Structure:")
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.LSTM):
                print(
                    f"  LSTM: input_size={module.input_size}, "
                    f"hidden_size={module.hidden_size}, "
                    f"num_layers={module.num_layers}, "
                    f"bidirectional={module.bidirectional}"
                )
            elif isinstance(module, torch.nn.Linear):
                print(
                    f"  Linear: in_features={module.in_features}, "
                    f"out_features={module.out_features}"
                )

    if hasattr(model, "decoder"):
        decoder = model.decoder
        print("Decoder Structure:")
        for name, module in decoder.named_modules():
            if isinstance(module, torch.nn.LSTM):
                print(
                    f"  LSTM: input_size={module.input_size}, "
                    f"hidden_size={module.hidden_size}, "
                    f"num_layers={module.num_layers}, "
                    f"bidirectional={module.bidirectional}"
                )
            elif isinstance(module, torch.nn.Linear):
                print(
                    f"  Linear: in_features={module.in_features}, "
                    f"out_features={module.out_features}"
                )


if __name__ == "__main__":

    input_dim = 1
    hidden_dim = config.LSTM_HIDDEN_DIM
    latent_dim = config.LATENT_DIM
    num_layers = config.LSTM_NUM_LAYERS
    beta = config.BETA

    model = LSTMBetaVAE(input_dim, hidden_dim, latent_dim, num_layers)

    print_model_parameters(model, model_name="LSTMBetaVAE")

    del model
    torch.cuda.empty_cache()
