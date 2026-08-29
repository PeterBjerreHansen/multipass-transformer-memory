from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES, allocate_blocks, normalized_weights


def test_dolmino_weights_are_normalized_from_published_50b_recipe():
    weights = normalized_weights()
    assert abs(sum(weights) - 1.0) < 1e-12
    raw = {source.name: source.weight for source in DOLMINO_50B_SOURCES}
    assert raw == {
        "dclm": 0.4720,
        "flan": 0.1660,
        "pes2o": 0.0585,
        "wiki": 0.0711,
        "stackexchange": 0.0245,
        "math": 0.2080,
    }


def test_block_allocation_is_exact_and_close_to_recipe():
    allocation = allocate_blocks(10_000)
    assert sum(allocation.values()) == 10_000
    normalized = normalized_weights()
    for source, expected in zip(DOLMINO_50B_SOURCES, normalized):
        assert abs(allocation[source.name] / 10_000 - expected) <= 1 / 10_000


def test_preparation_config_requires_exact_block_multiple():
    from tiny_mistral_mptt.data.config import DataPreparationConfig
    import pytest

    with pytest.raises(ValueError, match="multiples"):
        DataPreparationConfig(sequence_length=512, train_tokens=1000, validation_tokens=512).validate()

    with pytest.raises(ValueError, match="non-negative"):
        DataPreparationConfig(train_skip_tokens=-2048).validate()


def test_default_preparation_config_uses_active_2048_context():
    from tiny_mistral_mptt.data.config import DataPreparationConfig

    cfg = DataPreparationConfig()
    assert cfg.output_dir == "data/dolmino/wiring_2048"
    assert cfg.sequence_length == 2048
    assert cfg.train_tokens == 5_242_880
    assert cfg.validation_tokens == 524_288
    assert cfg.validation_skip_tokens == 0
    assert cfg.train_skip_tokens == 0
    assert cfg.shuffle_buffer == 25_000
    cfg.validate()
