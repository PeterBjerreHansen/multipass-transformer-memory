# Preserved feedback implementations

FBT and explicit memory-token attention are reference source, excluded from the
installed package and normal test suite. They are not supported by current
configuration, training, or evaluation commands.

The files below preserve their original paths. Memory-token behavior spans the
attention model, optional hybrid, dataset insertion and cached decoding; these
files are snapshots, not independently importable modules for the current code.
Matching tests and dependency declarations are included. Some mixed test files
also refer to other historical models; they are retained verbatim as evidence.

## Reproduce with the original runtime

Create a separate checkout at `5d4c2cd5a580974aba75b30c29971c71fc726308`. For example, from the repository root:

```bash
git worktree add --detach /tmp/feedback-reference 5d4c2cd5a580974aba75b30c29971c71fc726308
cd /tmp/feedback-reference
uv sync --extra data --extra eval
PYTHONPATH=src uv run pytest tests/test_fbt_variant.py tests/test_memory_attention_variant.py tests/test_memory_token_training_recovery.py
```

`SOURCE.json` records the revision and SHA-256 of each preserved file. Overlay
these snapshots at their original paths in the separate checkout if needed;
this preserves any pre-cleanup working-tree differences. Never overlay them
onto the active runtime. The archive contains no model weights or data.

The active implementation no longer supports MemoryAdd or the old middle-layer
Recirculation model. Their source is recoverable through Git history only.
