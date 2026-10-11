from count_lora_params import lora_params


def test_toy_shape_matches_the_hand_count():
    hidden, intermediate, layers, rank = 4, 8, 1, 8
    attn = 4 * rank * (hidden + hidden)
    mlp = 3 * rank * (hidden + intermediate)
    assert attn == 256
    assert mlp == 288
    assert lora_params(hidden, intermediate, layers, rank) == 544


def test_yaml_shape_doubles_when_rank_doubles():
    rank_8 = lora_params(2048, 8192, 8, 8)
    rank_16 = lora_params(2048, 8192, 8, 16)
    assert rank_16 == rank_8 * 2
