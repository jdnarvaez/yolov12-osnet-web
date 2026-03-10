"""
YoutuReID ONNX converter.

Inspects the raw YoutuReID models in raw_models/, adds a dynamic batch axis
on the first input/output dimension if it is fixed, and writes the patched
models to src/models/ so the web worker can load them via import.meta.url.

Requirements:
    pip install onnx onnxruntime

Run:
    python scripts/convert_youtureid.py

Models expected in raw_models/:
    person_reid_youtu_2021nov.onnx        (full FP32, ~106 MB)
    person_reid_youtu_2021nov_int8.onnx   (INT8 quantised, ~26 MB)
    person_reid_youtu_2021nov_int8bq.onnx (block-quantised INT8, ~29 MB)

Models written to src/models/:
    youtureid.onnx          ← int8 (recommended for web)
    youtureid_int8bq.onnx   ← block-quantised variant
    youtureid_fp32.onnx     ← full FP32 (large, included for completeness)
"""

from __future__ import annotations
from pathlib import Path
import sys

try:
    import onnx
    from onnx import TensorProto
except ImportError:
    sys.exit("onnx not found — run: pip install onnx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR    = REPO_ROOT / "raw_models"
OUT_DIR    = REPO_ROOT / "src" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONVERSIONS: list[tuple[str, str]] = [
    ("person_reid_youtu_2021nov_int8.onnx",   "youtureid.onnx"),
    ("person_reid_youtu_2021nov_int8bq.onnx", "youtureid_int8bq.onnx"),
    ("person_reid_youtu_2021nov.onnx",         "youtureid_fp32.onnx"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shape_str(vi: onnx.ValueInfoProto) -> str:
    """Return a human-readable shape string like [1, 3, 256, 128]."""
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return "[unknown]"
    parts = []
    for d in tt.shape.dim:
        if d.HasField("dim_param"):
            parts.append(d.dim_param)
        elif d.HasField("dim_value"):
            parts.append(str(d.dim_value))
        else:
            parts.append("?")
    return "[" + ", ".join(parts) + "]"


def _dtype_str(vi: onnx.ValueInfoProto) -> str:
    code = vi.type.tensor_type.elem_type
    return TensorProto.DataType.Name(code)


def make_batch_dynamic(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    """
    Ensure the first dimension of every graph input and output is the symbolic
    string 'batch'.  Returns the (possibly mutated) model and whether any
    change was made.
    """
    changed = False

    for vi in list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape") or len(tt.shape.dim) == 0:
            continue
        d = tt.shape.dim[0]
        if d.HasField("dim_param") and d.dim_param == "batch":
            continue          # already dynamic
        # Clear whatever was there and set the symbolic param
        d.ClearField("dim_value")
        d.dim_param = "batch"
        changed = True

    return model, changed


def inspect(model: onnx.ModelProto, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Opset : {model.opset_import[0].version}")
    for vi in model.graph.input:
        print(f"  Input : {vi.name:30s}  {_dtype_str(vi):10s}  {_shape_str(vi)}")
    for vi in model.graph.output:
        print(f"  Output: {vi.name:30s}  {_dtype_str(vi):10s}  {_shape_str(vi)}")


# ---------------------------------------------------------------------------
# Validate with ORT (optional)
# ---------------------------------------------------------------------------

def validate_ort(path: Path) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping ORT validation")
        return

    try:
        sess     = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        in_name  = sess.get_inputs()[0].name
        dummy    = np.random.randn(2, 3, 256, 128).astype(np.float32)
        out      = sess.run(None, {in_name: dummy})[0]
        print(f"  [ORT]  batch=2 → output shape {out.shape}  ✓")
    except Exception as e:
        print(f"  [ORT]  validation error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("YoutuReID ONNX converter")
print(f"  source : {RAW_DIR}")
print(f"  dest   : {OUT_DIR}")

for src_name, dst_name in CONVERSIONS:
    src_path = RAW_DIR / src_name
    dst_path = OUT_DIR / dst_name

    if not src_path.exists():
        print(f"\n[skip] {src_name} not found")
        continue

    print(f"\nProcessing {src_name} …")
    model = onnx.load(str(src_path))

    inspect(model, f"BEFORE  →  {src_name}")

    model, patched = make_batch_dynamic(model)
    if patched:
        print("  Patched: batch dimension is now dynamic")
    else:
        print("  Batch dimension was already dynamic — no change needed")

    inspect(model, f"AFTER   →  {dst_name}")

    onnx.checker.check_model(model)
    onnx.save(model, str(dst_path))

    size_mb = dst_path.stat().st_size / 1_048_576
    print(f"  Saved  {dst_path}  ({size_mb:.1f} MB)")

    validate_ort(dst_path)

print("\nDone.")
print()