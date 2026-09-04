def test_dolmino_packing_disables_persisted_padding_and_truncation():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from tiny_mistral_mptt.data.dolmino import configure_tokenizer_for_packing

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=1)

    configure_tokenizer_for_packing(tokenizer)

    assert tokenizer.padding is None
    assert tokenizer.truncation is None
    assert tokenizer.encode("hello hello", add_special_tokens=False).ids == [2, 2]


def test_serialized_tokenizer_settings_are_disabled_for_packing(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from tiny_mistral_mptt.data.dolmino import configure_tokenizer_for_packing

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(length=512, pad_id=1, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)
    serialized = tmp_path / "tokenizer.json"
    tokenizer.save(str(serialized))

    tokenizer = Tokenizer.from_file(str(serialized))
    assert tokenizer.padding is not None
    assert tokenizer.padding["pad_id"] == 1
    assert tokenizer.truncation is not None
    assert tokenizer.truncation["max_length"] == 512

    configure_tokenizer_for_packing(tokenizer)

    short = tokenizer.encode("hello world", add_special_tokens=False).ids
    long = tokenizer.encode("hello " * 1000, add_special_tokens=False).ids
    assert tokenizer.padding is None
    assert tokenizer.truncation is None
    assert 1 not in short
    assert len(short) < 512
    assert len(long) > 512
