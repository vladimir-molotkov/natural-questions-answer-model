import os

import torch


def configure_mps():
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.set_float32_matmul_precision("medium")
        print("MPS acceleration enabled")
    else:
        print("MPS not available, using CPU")
