"""Evaluator adapters for the shared Foundation Model: safetensors delta packages.

The pipeline, run from the repository root:

    python -m float_runtime.train_adapter ...      LoRA reward model on ModernBERT (+ reward_head.pt)
    python -m float_runtime.export_adapter ...     the safetensors delta package for a published foundation
    go run ./cmd/adapterbuild lora ...             the node's checks, keccak256 to register

One-time setup by the Telegraph team, not per adapter:

    python -m float_runtime.export_foundation ...  the Foundation Model ONNX with LoRA inputs
    bash float_runtime/build_engine.sh             the foundation engine (WASI reactor)

See README.md in this directory for the format and its determinism rules.
"""
