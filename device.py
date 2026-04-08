import torch

# Module-level flag set when DirectML is the active backend.
# BatchNorm2d is broken on DirectML — callers should use LayerNorm instead.
_DIRECTML_ACTIVE: bool = False


def get_device() -> torch.device:
    """Select best available device: CUDA > DirectML (AMD/Windows) > CPU."""
    global _DIRECTML_ACTIVE

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    try:
        import torch_directml
        device = torch_directml.device()
        _DIRECTML_ACTIVE = True
        print("Using DirectML (AMD GPU)  –  LayerNorm enabled (BatchNorm2d unsupported)")
        return device
    except Exception:
        pass

    print("Using CPU")
    return torch.device("cpu")


def is_directml() -> bool:
    """True when the active device is DirectML (BatchNorm2d unavailable)."""
    return _DIRECTML_ACTIVE
