.PHONY: test compile study-gates check download verify hf-check hf-layers hf-embeds \
  mps-smoke prepare-data verify-data evaluate-nll evaluate-quick substrate-gates \
  efficiency-mps efficiency-cuda efficiency-mps-training efficiency-mps-precision \
  efficiency-mps-context efficiency-mps-batch efficiency-cuda-training \
  efficiency-cuda-precision efficiency-cuda-context efficiency-cuda-batch \
  estimate-flops-frozen-backbone report-wiring-budgets cloud-preflight

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q src scripts tests
	uv run python -m py_compile scripts/start-and-watch scripts/run-cloud-study

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

estimate-flops-frozen-backbone:
	uv run python scripts/estimate_training_flops.py \
		--study benchmarks/development/frozen_backbone_comparison/STUDY.yaml \
		--model-config checkpoints/TinyMistral-248M-v3/config.json \
		--output benchmarks/development/frozen_backbone_comparison/results/training_flops.json

report-wiring-budgets:
	uv run python scripts/report_wiring_budgets.py \
		--study benchmarks/development/frozen_backbone_comparison/STUDY.yaml \
		--output benchmarks/development/frozen_backbone_comparison/results/wiring_budgets.json

# Usage: make cloud-preflight CONFIG=path/to/config.yaml
cloud-preflight:
	@test -n "$(CONFIG)" || (echo "CONFIG is required" && exit 2)
	uv run python scripts/cloud_preflight.py --config $(CONFIG)
