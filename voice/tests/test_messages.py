from messages import parse_client
import pytest


def test_each_client_type_parses():
    payloads = [
        {"type": "start"},
        {"type": "audio", "wav_b64": "aGVsbG8="},
        {"type": "pcm", "pcm16_b64": "aGVsbG8=", "sample_rate": 16000},
        {"type": "text", "text": "hello"},
        {"type": "barge_in"},
        {"type": "stop"},
    ]
    parsed = [parse_client(payload).type for payload in payloads]
    assert parsed == ["start", "audio", "pcm", "text", "barge_in", "stop"]


def test_unknown_client_type_is_rejected():
    with pytest.raises(ValueError):
        parse_client({"type": "nope"})
