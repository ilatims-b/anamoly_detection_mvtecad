import torch

# Load the checkpoint
ckpt_path = './results/padim/padim_model.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')

# List all keys in the model state dict
print("Checkpoint keys:", checkpoint.keys())
print("Model state dict keys:", checkpoint['model_state_dict'].keys())

# Check for 'mean' and 'icov' and print their shapes if present
state_dict = checkpoint['model_state_dict']
for key in ['mean', 'icov']:
    if key in state_dict:
        print(f"{key} found, shape: {state_dict[key].shape}")
    else:
        print(f"{key} NOT found in checkpoint.")