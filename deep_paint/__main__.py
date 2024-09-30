import torch

from deep_paint.lightning.cli import DeepLightningCLI


def main():
    """Driver script."""
    # Set torch precision depending on device
    device_name = torch.cuda.get_device_name()
    if "A100" or "H100" in device_name:
        torch.set_float32_matmul_precision("high")

    # CLI
    cli = DeepLightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"default_env": True}
    )

if __name__ == "__main__":
    main()
