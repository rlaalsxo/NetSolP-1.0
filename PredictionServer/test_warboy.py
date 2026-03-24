import onnx
from onnx import TensorProto, numpy_helper
from onnxsim import simplify
import numpy as np

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
print("simplify OK")


def cast_int64_to_int32(model):
    graph = model.graph
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.INT64:
            inp.type.tensor_type.elem_type = TensorProto.INT32
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.INT64:
            out.type.tensor_type.elem_type = TensorProto.INT32
    for init in graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
    for node in graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.INT64:
                attr.i = TensorProto.INT32
    return model


def cast_bool_to_float32(model):
    graph = model.graph
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.BOOL:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
    for init in graph.initializer:
        if init.data_type == TensorProto.BOOL:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
    for node in graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.BOOL:
                attr.i = TensorProto.FLOAT
    return model


simplified = cast_int64_to_int32(simplified)
simplified = cast_bool_to_float32(simplified)
print("type cast OK (int64->int32, bool->float32)")

onnx.save(simplified, OUTPUT_PATH)

from furiosa.runtime.sync import create_runner
try:
    runner = create_runner(OUTPUT_PATH)
    print("Warboy compile: SUCCESS")
    runner.close()
except Exception as e:
    print(f"Warboy compile: FAILED - {e}")
