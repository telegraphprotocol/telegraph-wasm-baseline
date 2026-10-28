# Runtime notes

## Current V2 model runtime

Telegraph V2 model adapters use a ModernBERT reward model exported to ONNX and
baked into a WebAssembly System Interface (WASI) command. The current package
contract is:

```text
ABI:       2
runtime:   onnx_f32
class:     judgment or hybrid
token cap: 1..512, default 512
```

The node provides a JSON input file containing canonical token IDs and an
all-ones attention mask. The guest reads that file and writes exactly one
finite FP32 score as raw bits to standard output. A non-zero exit, trap,
malformed output, multiple scores, NaN, or infinity rejects that score.

The guest is intentionally small:

- `export_onnx.py` merges a compatible LoRA reward checkpoint with the
  ModernBERT base model and writes the ONNX graph.
- `vm_float_wasm/bake_wasm.sh` compiles `vm_float_wasm/src/baked.rs` for
  `wasm32-wasip1`, embedding that graph at build time.
- `adapterbuild --baked-wasm` adds `telegraph.adapter` metadata and verifies
  the resulting WASI command before writing the final artifact and manifest.

See the repository [README](../README.md) for the supported command sequence.

## Runtime constraints

The node does not give an adapter ambient authority. The float-runtime
validator requires a WASI command with `_start`, its own memory, only the
allowed `wasi_snapshot_preview1` imports, and a valid `telegraph.adapter`
section. At execution, the node fixes the visible input file, command
arguments, clock, output limits, execution deadline, and memory limit.

An artifact's declared class must match the class of the target intent. A
`hybrid` intent still validates the model component through its model score;
the node applies the intent's comparator/fusion rules separately.

`max_tokens` is a resource and compatibility declaration, not a request to
the node to create longer inputs. The node truncates canonical tokens to that
limit, preserving the closing special token. Values above 512 are rejected.

## Validation boundary

The builder proves that the module is structurally packageable. The node is
the final authority for execution:

1. Registration validates metadata, WASI shape, imports, exports, size, and
   the target intent class.
2. Candidate evaluation compiles the module and runs a fixed smoke score.
3. A candidate that survives structural and smoke checks remains pending until
   compatible evaluation material exists, then competes using that material.

Do not treat a successful local package build as a production-quality or
ranking-quality result.

## Historical Q32.32 implementation

`wide_wasm`, `convert.py`, `convert_adapter.py`, and `.imwq` describe the
retired fixed-point model implementation. They are retained for forensic work
and fixture compatibility only. They are not the V2 model-adapter build path,
and a model adapter using the retired ABI is refused by V2 nodes.

The comparator-only WASM interface is separate from model adapters. It is
used by deterministic intent handling and does not turn a retired Q32.32 model
package into a supported V2 artifact.
