import torch


def acc_dtype(t: torch.Tensor) -> torch.dtype:
    """Accumulation dtype: float32 for low/single precision inputs, float64 for float64."""
    return torch.float64 if t.dtype == torch.float64 else torch.float32
