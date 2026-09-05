# Transformer memory experiments

The experiments compare how a pretrained transformer learns to use added feedback memory.

## Language

**Temporal Feedback**:
An autoregressive inference policy in which processing the current token consumes
latent state produced while processing earlier tokens.
_Avoid_: Recurrent inference as a generic name

**Live Feedback**:
Temporal Feedback after prompt prefill, using one evolving inference stream rather
than maintaining an exact stream for every refinement pass.
_Avoid_: Collapsed recurrence, recurrent continuation

**Fixed-route State Injection**:
Temporal Feedback in which each destination receives one predetermined preceding
record rather than selecting among retained records.
_Avoid_: Recurrence when the fixed access route is the intended distinction

**Parallel Multi-pass Training**:
A Jacobi-style training surrogate that exposes completed earlier-pass states to
later passes while keeping token computation parallel within each pass.
_Avoid_: Live recurrence, exact temporal replay

**Exact K-pass Continuation**:
Continuation that preserves all K refined inference streams and advances each
stream for every observed or generated token.
_Avoid_: Live Feedback

**Feedback Record**:
A latent representation emitted for possible use during later temporal feedback.
_Avoid_: Memory state when a particular retained record is intended

**Dense-and-strided Memory Attention**:
An attention-based feedback mechanism that reads recent dense memories and older,
regularly spaced memories together. The pure attention configuration has no
recurrent merger; adding the optional recurrent pathway makes it a hybrid.
_Avoid_: Multiscale Memory Attention (the former name), recurrent-attention hybrid

**Memory Attention**:
An added attention pathway that reads strict-past feedback records emitted by a
late-layer writer. The access pattern can be dense, strided, or dense-and-strided.
_Avoid_: Bank (the retired name)

**No-memory Adapter**:
A trainable capacity control that uses the current residual state but receives no
Feedback Record from an earlier token or pass.

**Recirculation-inspired Merger**:
A Fixed-route State Injection mechanism that uses adaptive, feature-wise mixing
borrowed from Recirculation without claiming to reproduce that paper's model.
_Avoid_: Recirculation reproduction, Recirculation baseline

**Dense Memory Attention**:
Memory Attention over the most recent ordinary-token memory records.

**Strided Memory Attention**:
Memory Attention over records written at fixed intervals in the actual input
sequence.

**Strided Self-Attention**:
Backbone self-attention over a local window and older regularly spaced keys.
It does not read cross-pass memory or add a memory reader.
_Avoid_: Strided Attention (ambiguous with Strided Memory Attention)

**Recurrent Memory / Memory Attention hybrid**:
A feedback model with both a preceding-token recurrent read and attention over
retained records, using the same late-emitted memory rule.
