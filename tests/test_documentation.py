"""Check current documentation interfaces without running experiments."""

from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from tiny_mistral_mptt.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
FENCES = re.compile(r"```([^\n]*)\n(.*?)```", re.S)


def documentation_files():
    return sorted({
        ROOT / "README.md", ROOT / "CONTEXT.md", ROOT / "data/README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "benchmarks").rglob("*.md"),
        *(ROOT / "evaluation").rglob("*.md"),
        ROOT / "scripts/README.md",
    })


def is_historical(path):
    relative = path.relative_to(ROOT).as_posix()
    return (
        relative.startswith(("benchmarks/historical/", "docs/research/"))
        or path.name == "FROZEN_WIRING_GRILL_EXCHANGE.md"
    )


def heading_anchors(content):
    counts = {}
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", content, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        index = counts.get(slug, 0)
        counts[slug] = index + 1
        anchors.add(slug if index == 0 else f"{slug}-{index}")
    return anchors


def test_documentation_local_links_and_heading_anchors_resolve():
    failures = []
    for path in documentation_files():
        content = FENCES.sub("", path.read_text(encoding="utf-8"))
        targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", content, re.M)
        for target in targets:
            target = target.split(' "', 1)[0].strip().strip("<>")
            if urlsplit(target).scheme:
                continue
            name, _, anchor = target.partition("#")
            name = unquote(name)
            destination = (
                ROOT / name.lstrip("/") if name.startswith("/")
                else path.parent / name if name else path
            )
            if not destination.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing {target}")
            elif anchor and destination.suffix == ".md":
                headings = heading_anchors(FENCES.sub("", destination.read_text(encoding="utf-8")))
                if unquote(anchor) not in headings:
                    failures.append(f"{path.relative_to(ROOT)}: missing anchor {target}")
    assert not failures, "\n".join(failures)


@lru_cache(maxsize=None)
def help_flags(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
        check=True, capture_output=True, text=True, timeout=30,
    )
    return set(re.findall(r"--[a-z][a-z0-9-]*", result.stdout))


def test_current_documented_cli_examples_use_existing_scripts_and_flags():
    for path in documentation_files():
        if is_historical(path):
            continue
        for language, block in FENCES.findall(path.read_text(encoding="utf-8")):
            if language.strip() not in {"bash", "sh", "shell"}:
                continue
            for line in block.replace("\\\n", " ").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                match = re.search(r"(?:\./)?(scripts/[\w.-]+)(?:\s|$)", line)
                if not match:
                    continue
                script = match.group(1)
                assert (ROOT / script).is_file(), (path, script)
                requested = set(re.findall(r"--[a-z][a-z0-9-]*", line[match.end():]))
                assert requested <= help_flags(script), (path, script, requested - help_flags(script))


@pytest.mark.parametrize("study", ["frozen_backbone_comparison", "frozen_backbone_lr_qualification"])
def test_documented_protocol_fields_match_every_declared_arm(study):
    directory = ROOT / "benchmarks/development" / study
    manifest = yaml.safe_load((directory / "STUDY.yaml").read_text())
    readme = (directory / "README.md").read_text()
    fields = re.findall(r"^\| `([a-z_]+)` \| `([^`]+)` \|$", readme, re.M)
    assert {name for name, _ in fields} >= {
        "phase", "max_unique_tokens", "eval_passes", "eval_batches",
        "eval_every_tokens", "feedback_eval_at_tokens",
    }
    for arm in manifest["arms"]:
        config = load_experiment_config(directory / arm["config"])
        for name, value in fields:
            assert getattr(config, name) == yaml.safe_load(value), (study, arm["id"], name)


def test_documented_snapshot_table_matches_optimizer_boundaries_and_selection():
    directory = ROOT / "benchmarks/development/frozen_backbone_comparison"
    config = load_experiment_config(directory / "recurrent_recirculation_multipass_100m.yaml")
    recipe = yaml.safe_load((ROOT / config.data_dir / "config.yaml").read_text())
    update_tokens = config.batch_size * config.grad_accum_steps * recipe["sequence_length"]
    rows = re.findall(
        r"^\| ([\d,]+) \| ([\d,]+) \| (\d+) \| (Yes|No) \|$",
        (directory / "README.md").read_text(), re.M,
    )
    assert [int(row[0].replace(",", "")) for row in rows] == config.snapshot_at_tokens
    for requested, actual, updates, feedback in rows:
        threshold = int(requested.replace(",", ""))
        steps = (threshold + update_tokens - 1) // update_tokens
        assert int(updates) == steps
        assert int(actual.replace(",", "")) == min(steps * update_tokens, config.max_unique_tokens)
        assert (feedback == "Yes") == (threshold in config.feedback_eval_at_tokens)
