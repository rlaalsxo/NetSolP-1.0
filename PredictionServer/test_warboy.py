import onnx
from onnxsim import simplify
from furiosa.runtime.sync import create_runner

MODEL_PATH = "models/Solubility_ESM1b_distilled_quantized.onnx"
OUTPUT_PATH = "models/test_static.onnx"

model = onnx.load(MODEL_PATH)
simplified, ok = simplify(
    model,
    overwrite_input_shapes={
        "tokens": [1, 1024],
        "lengths": [1],
        "non_pad_mask": [1, 1024],
    },
)
if not ok:
    print("simplify failed")
    exit(1)
onnx.save(simplified, OUTPUT_PATH)
print("simplify OK")

try:
    runner = create_runner(OUTPUT_PATH)
    print("Warboy compile: SUCCESS")
    runner.close()
except Exception as e:
    print(f"Warboy compile: FAILED - {e}")
