"""Regression tests for the export guards. Run:
    python -m wide_runtime.test_export_onnx
Needs only numpy and onnx (no torch, no model download).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .export_onnx import check_self_contained, legacy_export_kwargs


def tiny_model() -> onnx.ModelProto:
    w = numpy_helper.from_array(np.ones((4, 1), dtype=np.float32), name="w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_single_file_passes():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "model.onnx")
        onnx.save_model(tiny_model(), str(p))
        check_self_contained(onnx.load(str(p), load_external_data=False), p)


def test_external_data_is_refused():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "model.onnx")
        onnx.save_model(tiny_model(), str(p), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=0)
        assert Path(d, "model.onnx.data").exists()
        try:
            check_self_contained(onnx.load(str(p), load_external_data=False), p)
        except SystemExit as e:
            assert "external data" in str(e)
        else:
            raise AssertionError("a split ONNX model passed the self-contained check")


def test_legacy_export_kwargs():
    def new_export(model, args, f, *, dynamo=True): ...
    def old_export(model, args, f): ...
    assert legacy_export_kwargs(new_export) == {"dynamo": False}
    assert legacy_export_kwargs(old_export) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
