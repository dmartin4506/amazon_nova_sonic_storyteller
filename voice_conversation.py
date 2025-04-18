import asyncio
import base64
import json
import uuid
import pyaudio

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

# Audio settings
INPUT_SAMPLE_RATE  = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS           = 1
FORMAT             = pyaudio.paInt16
CHUNK_SIZE         = 1024

class NovaSonicVoiceAssistant:
    """Keeps one long‐lived bidirectional stream open for voice Q&A."""

    def __init__(self,
                 model_id="amazon.nova-sonic-v1:0",
                 region="us-east-1"):
        self.model_id         = model_id
        self.region           = region
        self.prompt_id        = str(uuid.uuid4())
        self.text_content_id  = str(uuid.uuid4())
        self.audio_content_id = str(uuid.uuid4())
        self.client           = None
        self.stream           = None
        self.is_active        = False
        self.audio_queue      = asyncio.Queue()

    async def start_session(self):
        # 1) Initialize Bedrock client
        cfg = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.client = BedrockRuntimeClient(config=cfg)

        # 2) Open bidi stream
        resp = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.stream = resp
        self.is_active = True

        # 3) Send sessionStart
        await self._send_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7
                    }
                }
            }
        })

        # 4) Send promptStart
        await self._send_event({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_id,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": CHANNELS,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": []}
                }
            }
        })

        # 5) Send story as SYSTEM turn
        from story_narration import story_text
        # contentStart
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_id,
                    "contentName": self.text_content_id,
                    "role": "SYSTEM",
                    "type": "TEXT",
                    "interactive": True,
                    "textInputConfiguration": {"mediaType":"text/plain"}
                }
            }
        })
        # textInput
        await self._send_event({
            "event": {
                "textInput": {
                    "promptName": self.prompt_id,
                    "contentName": self.text_content_id,
                    "content": story_text
                }
            }
        })
        # contentEnd
        await self._send_event({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_id,
                    "contentName": self.text_content_id
                }
            }
        })

        # 6) Start reader task
        asyncio.create_task(self._receive_events())

    async def _send_event(self, evt: dict):
        """Wrap JSON in a chunk and send it into the stream."""
        part = BidirectionalInputPayloadPart(
            bytes_=json.dumps(evt).encode("utf-8")
        )
        chunk = InvokeModelWithBidirectionalStreamInputChunk(value=part)
        await self.stream.input_stream.send(chunk)

    async def _receive_events(self):
        """Continuously read Nova Sonic’s responses and queue the PCM data."""
        while self.is_active:
            try:
                _, operation = await self.stream.await_output()
                msg = await operation.receive()
            except Exception:
                break

            if msg.value and msg.value.bytes_:
                data = json.loads(msg.value.bytes_.decode("utf-8"))
                evt = data.get("event", {})
                if "audioOutput" in evt:
                    pcm = base64.b64decode(evt["audioOutput"]["content"])
                    await self.audio_queue.put(pcm)
                if "contentEnd" in evt and evt["contentEnd"].get("contentName"):
                    # sentinel for end of turn
                    await self.audio_queue.put(None)

        # final sentinel
        await self.audio_queue.put(None)

    async def capture_audio(self):
        """Continuously capture mic audio and send it as one audio turn."""
        pa = pyaudio.PyAudio()
        mic = pa.open(format=FORMAT, channels=CHANNELS,
                      rate=INPUT_SAMPLE_RATE, input=True,
                      frames_per_buffer=CHUNK_SIZE)

        # signal start of user audio
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_id,
                    "contentName": self.audio_content_id,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": CHANNELS,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }
                }
            }
        })

        try:
            while self.is_active:
                data = mic.read(CHUNK_SIZE, exception_on_overflow=False)
                b64  = base64.b64encode(data).decode("utf-8")
                await self._send_event({
                    "event": {
                        "audioInput": {
                            "promptName": self.prompt_id,
                            "contentName": self.audio_content_id,
                            "content": b64
                        }
                    }
                })
                await asyncio.sleep(0)  # yield to other tasks
        finally:
            mic.stop_stream()
            mic.close()
            pa.terminate()
            # signal end of user audio
            await self._send_event({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_id,
                        "contentName": self.audio_content_id
                    }
                }
            })

    async def play_audio(self):
        """Continuously dequeue and play back Nova Sonic’s PCM responses."""
        pa  = pyaudio.PyAudio()
        out = pa.open(format=FORMAT, channels=CHANNELS,
                      rate=OUTPUT_SAMPLE_RATE, output=True)
        while self.is_active:
            pcm = await self.audio_queue.get()
            if pcm is None:
                break
            out.write(pcm)
        out.stop_stream()
        out.close()
        pa.terminate()

    async def close(self):
        """Gracefully close the Nova Sonic session."""
        self.is_active = False
        # sessionEnd
        await self._send_event({"event":{"sessionEnd":{}}})
        # close the stream
        await self.stream.input_stream.end()
        await self.client.close()
