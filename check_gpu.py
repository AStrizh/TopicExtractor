import torch

# Checks if GPU dependencies were properly installed. Should return your GPU name.
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available.")
