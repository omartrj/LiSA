import torch

def diagnose_torch():
    # Check torch version and if it is a cpu-only build
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA compiled in PyTorch: {torch.version.cuda}")

if __name__ == "__main__":
    diagnose_torch()