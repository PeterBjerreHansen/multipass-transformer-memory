# Memory Attention

Memory Attention is the public name for the model family that exposes
previous-pass representations as a separately addressable key/value source.
The mechanism is **cross-pass attention**: each current-pass token selects
from the causally valid memory records produced by an earlier pass.

## Why the name changed

“Bank” described the storage metaphor, but not the operation that makes these
models distinct. The research vocabulary is now:

- **Memory Attention** — the architecture family.
- **cross-pass attention** — the mechanism.
- **memory records/states** — stored previous-pass representations.
- **memory-attention reader** — the GQA cross-attention module.
- **dense, strided, memory-token, and multiscale** — access patterns.

The implementation retains historical `bank` and `periodic` identifiers in
Python symbols, serialized fields, configuration filenames, checkpoints, and
result paths. Those names are part of the reproducibility contract. New configs may use
`memory_attention`, `memory_attention_multiscale`,
`memory_attention_add_hybrid`, and
`memory_attention_recirculation_hybrid`; they are compatibility aliases for
the corresponding historical variants.

## Model names

| Public model name | Historical config alias | Access pattern |
| --- | --- | --- |
| Dense Memory Attention | `bank` | recent previous-pass states |
| Sparse Strided Memory Attention | `bank` + `memory_write_mode: periodic` | states retained at a fixed stride |
| Sparse Memory-token Attention | `bank` + `memory_write_mode: memory_token` | explicit `<MEM>` records |
| Multiscale Memory Attention | `bank_multiscale` | dense recent + sparse older states |
| Recirculation + Sparse Memory Attention | `bank_recirculation_hybrid` | fast recurrence + sparse records |

The compatibility contract, masks, write timing, and cached-inference details
remain documented in [BANK_MEMORY.md](BANK_MEMORY.md). The architecture map is
in [ARCHITECTURES.md](ARCHITECTURES.md).
