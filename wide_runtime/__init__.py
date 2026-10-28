"""Telegraph V2 scoring-adapter tools.

The supported build path trains a ModernBERT LoRA reward model, exports it to
ONNX, bakes the graph into a restricted WASI command, and packages it as an
ABI-v2 ``onnx_f32`` adapter. The older Q32.32 modules remain only for legacy
fixtures and reference work.
"""
