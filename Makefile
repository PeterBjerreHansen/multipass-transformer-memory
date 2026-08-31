.PHONY: test compile study-gates check download verify hf-check hf-layers hf-embeds \
  mps-smoke prepare-data verify-data evaluate-nll evaluate-quick substrate-gates \
  efficiency-mps efficiency-cuda efficiency-mps-training efficiency-mps-precision \
  efficiency-mps-context efficiency-mps-batch efficiency-cuda-training \
  efficiency-cuda-precision efficiency-cuda-context efficiency-cuda-batch \
  efficiency-cuda-stage5 \
  efficiency-cuda-forward-modes \
  estimate-flops-stage5 estimate-flops-forward-modes \
  efficiency-cuda-batch-qualification efficiency-bank-write \
  select-cuda-batch cloud-preflight

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q src scripts tests
	uv run python -m py_compile scripts/start-and-watch scripts/run-cloud-campaign

study-gates:
	uv run python scripts/verify_study.py

check: test compile study-gates
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git diff --check; \
	else echo "SKIP: git diff --check (no Git metadata in this source snapshot)"; fi

download:
	uv run python scripts/download_model.py

verify:
	uv run python scripts/verify_model.py

hf-check:
	uv run python scripts/compare_to_hf.py --device cpu --dtype float32

hf-layers:
	uv run python scripts/compare_to_hf_layers.py --device cpu --dtype float32

hf-embeds:
	uv run python scripts/compare_to_hf_inputs_embeds.py --length 40

mps-smoke:
	uv run python scripts/smoke_mps.py

prepare-data:
	uv run python scripts/prepare_data.py

verify-data:
	uv run python scripts/verify_data.py data/dolmino/wiring_2048

evaluate-nll:
	uv run python scripts/evaluate_nll.py \
		--config benchmarks/controls/substrate/mac.yaml \
		--initialized-baseline --passes 1

evaluate-quick:
	uv run python scripts/evaluate_lm_harness.py \
		--config benchmarks/controls/substrate/mac.yaml \
		--suite evaluation/suites/quick.yaml --limit 100 \
		--initialized-baseline --prefill-passes 1 --decode-mode standard

substrate-gates: check verify hf-check hf-layers hf-embeds mps-smoke

# Engineering-only efficiency characterization. These compact JSON results are retained.
efficiency-mps: efficiency-mps-training efficiency-mps-precision efficiency-mps-context efficiency-mps-batch

efficiency-cuda: efficiency-cuda-training efficiency-cuda-precision efficiency-cuda-context efficiency-cuda-batch

efficiency-mps-training:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/training.yaml --device mps \
		--output benchmarks/efficiency/results/mps_training.json

efficiency-mps-precision:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/precision_mps.yaml \
		--output benchmarks/efficiency/results/mps_precision.json

efficiency-mps-context:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/context_scaling.yaml --device mps \
		--output benchmarks/efficiency/results/mps_context.json

efficiency-mps-batch:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/batch_scaling.yaml --device mps \
		--output benchmarks/efficiency/results/mps_batch.json

efficiency-cuda-training:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/training.yaml --device cuda \
		--output benchmarks/efficiency/results/cuda_training.json

efficiency-cuda-precision:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/precision_cuda.yaml \
		--output benchmarks/efficiency/results/cuda_precision.json

efficiency-cuda-context:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/context_scaling.yaml --device cuda --autocast-dtype bfloat16 \
		--output benchmarks/efficiency/results/cuda_context.json

efficiency-cuda-batch:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/batch_scaling.yaml --device cuda --autocast-dtype bfloat16 \
		--output benchmarks/efficiency/results/cuda_batch.json

efficiency-cuda-stage5:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/stage_5_architectures.yaml \
		--output benchmarks/efficiency/results/stage_5_architectures.json

efficiency-cuda-forward-modes:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/forward_modes.yaml \
		--output benchmarks/efficiency/results/cuda_forward_modes.json

estimate-flops-stage5:
	uv run python scripts/estimate_training_flops.py \
		--suite benchmarks/efficiency/suites/stage_5_architectures.yaml \
		--model-config checkpoints/TinyMistral-248M-v3/config.json \
		--output benchmarks/efficiency/results/stage_5_training_flops.json

estimate-flops-forward-modes:
	uv run python scripts/estimate_training_flops.py \
		--suite benchmarks/efficiency/suites/forward_modes.yaml \
		--schedule 2:1 \
		--model-config checkpoints/TinyMistral-248M-v3/config.json \
		--output benchmarks/efficiency/results/forward_mode_training_flops.json


# Engineering-only bank write-cadence scaling. This does not select C.
efficiency-bank-write:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/bank_write_scaling.yaml \
		--device cuda --autocast-dtype bfloat16 \
		--output benchmarks/efficiency/results/bank_write_scaling.json

# Core-run batching qualification. This keeps grad accumulation at 1 so the
# hardware microbatch axis is measured without silently changing optimizer batch.
efficiency-cuda-batch-qualification:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite benchmarks/efficiency/suites/cuda_batch_qualification.yaml \
		--output benchmarks/efficiency/results/cuda_batch_qualification.json

# Usage: make select-cuda-batch RESULT=benchmarks/efficiency/results/cuda_batch_qualification.json
select-cuda-batch:
	@test -n "$(RESULT)" || (echo "RESULT is required" && exit 2)
	uv run python scripts/select_cuda_batch.py $(RESULT)

# Usage: make cloud-preflight CONFIG=path/to/config.yaml
cloud-preflight:
	@test -n "$(CONFIG)" || (echo "CONFIG is required" && exit 2)
	uv run python scripts/cloud_preflight.py --config $(CONFIG)
