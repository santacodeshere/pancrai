import sys
sys.path.insert(0, '.')
import torch
from app.models.transunet import build_transunet

print("Loading model...")
model = build_transunet(weights_path='./weights/transunet_best.pth')
model.eval()

print("\n=== All Conv2d layers ===")
conv_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_layers.append((name, module))
        print(f"  {name:60s} kernel={module.kernel_size} out_ch={module.out_channels}")

print(f"\nTotal Conv2d layers: {len(conv_layers)}")
print(f"\nLast 10 layers:")
for name, module in conv_layers[-10:]:
    print(f"  {name:60s} kernel={module.kernel_size} out_ch={module.out_channels}")

# Find best candidate - largest spatial conv near the end
print("\n=== Best candidates (non 1x1, last 5) ===")
non_1x1 = [(n, m) for n, m in conv_layers
           if m.kernel_size not in [(1,1), 1]]
for name, module in non_1x1[-5:]:
    print(f"  {name:60s} kernel={module.kernel_size} out_ch={module.out_channels}")
