from segment import SentenceBuffer


def test_sentence_completes_across_two_pushes():
    buffer = SentenceBuffer()
    assert buffer.push("Hello") == []
    assert buffer.push(" there.") == ["Hello there."]


def test_command_name_stays_one_sentence():
    buffer = SentenceBuffer()
    assert buffer.push("mlx_lm.lora is a command.") == ["mlx_lm.lora is a command."]


def test_decimal_stays_one_sentence():
    buffer = SentenceBuffer()
    assert buffer.push("1.5 seconds.") == ["1.5 seconds."]
    assert buffer.flush() == ""
