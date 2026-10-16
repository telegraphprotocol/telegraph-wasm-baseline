"""Wide Q32.32 deterministic inference pipeline.

Ported from intent_training_experiments/pipelines/wide_runtime. Converts a float HF encoder
(ModernBERT) plus a LoRA adapter into a `.imwq` artifact whose weights AND activations are
Q32.32 fixed-point int64 — the format the Rust runtime in `wide_wasm/` and the Go host in
`pkg/wasm/runtime/wide.go` execute.

Why wide rather than int8: int8 requantises activations to 256 levels between every layer,
and the reward signal separating a good answer from a bad one is smaller than the rounding
error accumulated across ~22 layers (measured Spearman vs float ~0.001 — no better than
shuffling the miners). Q32.32 keeps the signal: op-level error against float is ~3e-10.
It is still fully integer, so validators compute bit-identical scores and the epoch cannot
fork on floating-point rounding.
"""
