# Transformer memory experiments

The experiments compare how a pretrained transformer learns to use added feedback memory.

## Language

**Dense-and-strided Memory Attention**:
An attention-based feedback mechanism that reads recent dense memories and older,
regularly spaced memories together. The pure attention configuration has no
recurrent merger; adding the optional recurrent pathway makes it a hybrid.
_Avoid_: Multiscale Memory Attention (the former name), recurrent-attention hybrid

**Memory Attention**:
An added attention pathway that reads strict-past feedback records emitted by a
late-layer writer. The access pattern can be dense, strided, or dense-and-strided.
_Avoid_: Bank (the retired name)

**Dense Memory Attention**:
Memory Attention over the most recent ordinary-token memory records.

**Strided Memory Attention**:
Memory Attention over regularly spaced ordinary-token memory records.

**Strided Self-Attention**:
Backbone self-attention over a local window and older regularly spaced keys.
It does not read cross-pass memory or add a memory reader.
_Avoid_: Strided Attention (ambiguous with Strided Memory Attention)

**Recurrent Memory / Memory Attention hybrid**:
A feedback model with both a preceding-token recurrent read and attention over
retained records, using the same late-emitted memory rule.
