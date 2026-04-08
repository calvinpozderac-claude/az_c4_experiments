import torch


def get_device() -> torch.device:
    """Select best available device: CUDA > DirectML (AMD/Windows) > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML (AMD GPU)")
        return device
    except Exception:
        pass

    print("Using CPU")
    return torch.device("cpu")
