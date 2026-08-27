# Memory Attention

Memory Attention is the public name for the model family that exposes
previous-pass representations as a separately addressable key/value source.
The mechanism is **cross-pass attention**: each current-pass token selects
from the causally valid memory records produced by an earlier pass.

## Model names

| Model name | Access pattern |
| --- | --- |
| Dense Memory Attention | recent previous-pass states |
| Strided Memory Attention | regularly strided previous-pass states |
| Memory-token Attention | explicit `<MEM>` states |
| Multiscale Memory Attention | dense recent + sparse older states |
| Recirculation + Strided Memory Attention | fast recurrence + strided records |

The masks, write timing, and cached-inference details remain documented in
[BANK_MEMORY.md](BANK_MEMORY.md). The architecture map is in
[ARCHITECTURES.md](ARCHITECTURES.md).
