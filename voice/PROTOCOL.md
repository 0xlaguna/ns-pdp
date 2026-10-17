# Voice WebSocket

One socket at `ws://127.0.0.1:8010/ws`.

Client to server:

| type | fields |
| --- | --- |
| `start` | |
| `audio` | `wav_b64` |
| `pcm` | `pcm16_b64`, `sample_rate` |
| `text` | `text` |
| `barge_in` | |
| `stop` | |

Server to client:

| type | fields |
| --- | --- |
| `transcript` | `text` |
| `token` | `text` |
| `audio` | `wav_b64` |
| `cancel_audio` | |
| `turn_done` | |
| `error` | `message` |

`barge_in` cancels the current turn. The server then sends `cancel_audio` and no further `token` or `audio` for that turn. `error` names a failure and leaves the process up.
