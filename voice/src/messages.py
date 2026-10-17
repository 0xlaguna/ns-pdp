"""Pydantic messages for the voice socket. Discriminator is ``type``."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class Start(BaseModel):
    type: Literal["start"]


class AudioIn(BaseModel):
    type: Literal["audio"]
    wav_b64: str


class Pcm(BaseModel):
    type: Literal["pcm"]
    pcm16_b64: str
    sample_rate: int


class TextIn(BaseModel):
    type: Literal["text"]
    text: str


class BargeIn(BaseModel):
    type: Literal["barge_in"]


class Stop(BaseModel):
    type: Literal["stop"]


class Transcript(BaseModel):
    type: Literal["transcript"]
    text: str


class Token(BaseModel):
    type: Literal["token"]
    text: str


class AudioOut(BaseModel):
    type: Literal["audio"]
    wav_b64: str


class CancelAudio(BaseModel):
    type: Literal["cancel_audio"]


class TurnDone(BaseModel):
    type: Literal["turn_done"]


class ErrorMessage(BaseModel):
    type: Literal["error"]
    message: str


ClientMessage = Annotated[
    Union[Start, AudioIn, Pcm, TextIn, BargeIn, Stop],
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    Union[Transcript, Token, AudioOut, CancelAudio, TurnDone, ErrorMessage],
    Field(discriminator="type"),
]

_client = TypeAdapter(ClientMessage)
_server = TypeAdapter(ServerMessage)


def parse_client(payload: dict) -> Start | AudioIn | Pcm | TextIn | BargeIn | Stop:
    try:
        return _client.validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"unknown client message: {payload!r}") from exc


def parse_server(payload: dict):
    try:
        return _server.validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"unknown server message: {payload!r}") from exc
