from ifd.errors import ForgeryError, model_load_error


def test_model_load_error_adds_ram_hint_only_for_memory_failures():
    e = model_load_error("Mesorch", "m.onnx", RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation"))
    assert isinstance(e, ForgeryError) and "thiếu RAM" in str(e) and "m.onnx" in str(e)
    e = model_load_error("Mesorch", "m.onnx", MemoryError())
    assert "MemoryError" in str(e) and "thiếu RAM" in str(e)
    e = model_load_error("Mesorch", "m.onnx", RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model failed"))
    assert "thiếu RAM" not in str(e) and "INVALID_PROTOBUF" in str(e)
