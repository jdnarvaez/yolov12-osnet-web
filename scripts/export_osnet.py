"""
Self-contained OSNet x0.25 → ONNX exporter.

No torchreid required. The architecture is defined inline; pretrained weights
are downloaded directly from the original Google Drive link.

Requirements (already installed in .venv):
    pip install torch gdown

Run:
    python scripts/export_osnet.py

Output:
    src/models/osnet_x0_25.onnx  (~1.5 MB)

Model I/O
---------
    input  float32  [N, 3, 256, 128]   N crops, CHW, ImageNet-normalised
    output float32  [N, 512]            L2-normalised embedding per crop
"""

from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("torch not found — run: pip install torch")

try:
    import gdown
except ImportError:
    sys.exit("gdown not found — run: pip install gdown")

# ---------------------------------------------------------------------------
# OSNet architecture (source: KaiyangZhou/deep-person-reid, MIT licence)
# Only the layers needed for inference (eval mode) are included.
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Conv1x1Linear(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False, groups=out_c)
        self.bn    = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))

class ChannelGate(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(in_c, in_c // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Conv2d(in_c // reduction, in_c, 1, bias=True)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc2(self.relu(self.fc1(self.pool(x)))))

class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, bottleneck_reduction=4):
        super().__init__()
        mid = out_c // bottleneck_reduction
        self.conv1  = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate   = ChannelGate(mid)
        self.conv3      = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None
    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)

def _make_layer(block, layer, in_c, out_c, reduce_spatial):
    layers = [block(in_c, out_c)]
    for _ in range(1, layer):
        layers.append(block(out_c, out_c))
    if reduce_spatial:
        layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
    return nn.Sequential(*layers)

class OSNet_x0_25(nn.Module):
    """OSNet width×0.25 — channels [16, 64, 96, 128], feature_dim=512."""
    def __init__(self):
        super().__init__()
        ch = [16, 64, 96, 128]
        self.conv1      = ConvLayer(3, ch[0], 7, stride=2, padding=3)
        self.maxpool    = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2      = _make_layer(OSBlock, 2, ch[0], ch[1], reduce_spatial=True)
        self.conv3      = _make_layer(OSBlock, 2, ch[1], ch[2], reduce_spatial=True)
        self.conv4      = _make_layer(OSBlock, 2, ch[2], ch[3], reduce_spatial=False)
        self.conv5      = Conv1x1(ch[3], ch[3])
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Sequential(nn.Linear(ch[3], 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avg(x).view(x.size(0), -1)
        return self.fc(v)   # [N, 512]

# ---------------------------------------------------------------------------
# Download pretrained weights
# ---------------------------------------------------------------------------
WEIGHTS_URL = "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs"
WEIGHTS_PATH = Path("osnet_x0_25_imagenet.pth")

if not WEIGHTS_PATH.exists():
    print(f"Downloading pretrained weights → {WEIGHTS_PATH} …")
    gdown.download(WEIGHTS_URL, str(WEIGHTS_PATH), quiet=False)
else:
    print(f"Found cached weights at {WEIGHTS_PATH}")

# ---------------------------------------------------------------------------
# Load weights
# ---------------------------------------------------------------------------
model = OSNet_x0_25()

state = torch.load(str(WEIGHTS_PATH), map_location="cpu")

# The checkpoint may be a raw state_dict or wrapped in {'state_dict': ...}
if "state_dict" in state:
    state = state["state_dict"]

# Strip any 'module.' prefix (DataParallel checkpoints)
state = {k.replace("module.", ""): v for k, v in state.items()}

# Only load keys that match in name and shape; skip the classifier head
model_state = model.state_dict()
matched   = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
unmatched = [k for k in state if k not in matched]

model.load_state_dict(matched, strict=False)
print(f"Loaded {len(matched)} / {len(state)} weight tensors  "
      f"(skipped: {unmatched[:5]}{'…' if len(unmatched) > 5 else ''})")

model.eval()

# Sanity-check
with torch.no_grad():
    dummy = torch.zeros(2, 3, 256, 128)
    out   = model(dummy)
    assert out.shape == (2, 512), f"Unexpected output shape: {out.shape}"
print(f"Forward pass OK — output shape: {out.shape}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
out_dir  = Path(__file__).parent.parent / "src" / "models"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "osnet_x0_25.onnx"

print(f"Exporting to {out_path} …")
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        do_constant_folding=True,
    )

size_mb = out_path.stat().st_size / 1_048_576
print(f"Exported  {out_path}  ({size_mb:.1f} MB)")

# ---------------------------------------------------------------------------
# Optional ORT validation
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    import numpy as np
    sess   = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp    = np.random.randn(3, 3, 256, 128).astype(np.float32)
    result = sess.run(None, {"input": inp})[0]
    assert result.shape == (3, 512)
    print(f"ORT validation: batch=3 → {result.shape}  ✓")
except ImportError:
    print("onnxruntime not installed — skipping ORT validation.")
except Exception as e:
    print(f"ORT validation error: {e}")
