"""Close sentences in a token stream without splitting commands or decimals."""

from __future__ import annotations


def split_complete(buf: str) -> tuple[list[str], str]:
    ready: list[str] = []
    start = 0
    index = 0
    while index < len(buf):
        char = buf[index]
        if char in ".?!":
            previous = buf[index - 1] if index > 0 else ""
            nxt = buf[index + 1] if index + 1 < len(buf) else ""
            if char == "." and previous.isalnum() and nxt.isalnum():
                index += 1
                continue
            if char == "." and previous.isdigit():
                index += 1
                continue
            if nxt == "" or nxt.isspace():
                sentence = buf[start : index + 1].strip()
                if sentence:
                    ready.append(sentence)
                start = index + 1
                while start < len(buf) and buf[start].isspace():
                    start += 1
                index = start
                continue
        index += 1
    return ready, buf[start:]


class SentenceBuffer:
    def __init__(self) -> None:
        self.buf = ""

    def push(self, token: str) -> list[str]:
        self.buf += token
        ready, self.buf = split_complete(self.buf)
        return ready

    def flush(self) -> str:
        text = self.buf.strip()
        self.buf = ""
        return text
