import os
import asyncio
import base64
import json
import uuid

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

import boto3
import pyaudio
from dotenv import load_dotenv

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

load_dotenv()

# Audio parameters
INPUT_SAMPLE_RATE  = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS           = 1
FORMAT             = pyaudio.paInt16
CHUNK_SIZE         = 1024

# Story text
STORY = (
    "Once upon a time in a quiet village, there lived a wise old storyteller. "
    "Every evening, children gathered to hear tales of adventure and magic. "
    "One night, the storyteller began a new tale about a hidden treasure in the forest, "
    "protected by an ancient talking tree. Little did the villagers know, this story "
    "would soon come to life, and they would become part of the adventure."
)

#——– Nova Sonic session manager ————————————————
class NovaSonicSession:
    def __init__(self, model_id="amazon.nova-sonic-v1:0", region="us-east-1"):
        self.prompt_id        = str(uuid.uuid4())
        self.text_content_id  = str(uuid.uuid4())
        self.audio_content_id = str(uuid.uuid4())
        self.is_active        = False
        self.audio_queue      = asyncio.Queue()
        # init client
        cfg = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.client = BedrockRuntimeClient(config=cfg)

    async def start(self):
        # open bidi stream
        resp = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id="amazon.nova-sonic-v1:0")
        )
        self.stream = resp
        self.is_active = True

        # send sessionStart, promptStart
        await self._send_json({
            "event": {
                "sessionStart": {"inferenceConfiguration": {"maxTokens":1024,"topP":0.9,"temperature":0.7}}
            }
        })
        await self._send_json({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_id,
                    "textOutputConfiguration":{"mediaType":"text/plain"},
                    "audioOutputConfiguration":{
                        "mediaType":"audio/lpcm",
                        "sampleRateHertz":OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits":16,
                        "channelCount":CHANNELS,
                        "voiceId":"matthew",
                        "encoding":"base64",
                        "audioType":"SPEECH"
                    },
                    "toolUseOutputConfiguration":{"mediaType":"application/json"},
                    "toolConfiguration":{"tools":[]}
                }
            }
        })

        # send story as SYSTEM turn
        for evt in [
            {"event":{"contentStart":{"promptName":self.prompt_id,"contentName":self.text_content_id,"role":"SYSTEM","type":"TEXT","interactive":True,"textInputConfiguration":{"mediaType":"text/plain"}}}},
            {"event":{"textInput":{"promptName":self.prompt_id,"contentName":self.text_content_id,"content":STORY}}},
            {"event":{"contentEnd":{"promptName":self.prompt_id,"contentName":self.text_content_id}}}
        ]:
            await self._send_json(evt)

        # start reader
        asyncio.create_task(self._reader())

    async def _send_json(self, evt: dict):
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(evt).encode())
        )
        await self.stream.input_stream.send(chunk)

    async def _reader(self):
        while self.is_active:
            _, op = await self.stream.await_output()
            msg = await op.receive()
            if msg.value and msg.value.bytes_:
                data = json.loads(msg.value.bytes_.decode())
                if "audioOutput" in data["event"]:
                    pcm = base64.b64decode(data["event"]["audioOutput"]["content"])
                    await self.audio_queue.put(pcm)
        # sentinel
        await self.audio_queue.put(None)

    async def send_audio(self, pcm_bytes: bytes):
        # send contentStart only once at beginning of turn
        await self._send_json({
            "event":{
                "contentStart":{
                    "promptName":self.prompt_id,
                    "contentName":self.audio_content_id,
                    "type":"AUDIO","interactive":True,"role":"USER",
                    "audioInputConfiguration":{
                        "mediaType":"audio/lpcm","sampleRateHertz":INPUT_SAMPLE_RATE,
                        "sampleSizeBits":16,"channelCount":CHANNELS,"audioType":"SPEECH","encoding":"base64"
                    }
                }
            }
        })
        # send audioInput chunk
        await self._send_json({
            "event":{
                "audioInput":{
                    "promptName":self.prompt_id,
                    "contentName":self.audio_content_id,
                    "content":base64.b64encode(pcm_bytes).decode()
                }
            }
        })
        # end turn
        await self._send_json({
            "event":{
                "contentEnd":{
                    "promptName":self.prompt_id,
                    "contentName":self.audio_content_id
                }
            }
        })

        # rotate content ID for next question
        self.audio_content_id = str(uuid.uuid4())

    async def close(self):
        self.is_active = False
        await self._send_json({"event":{"sessionEnd":{}}})
        await self.stream.input_stream.end()
        await self.client.close()


#——– Streamlit UI ——————————————————————————————————————————

st.title("📖 Story & Nova Sonic Q&A")

# 1) Narrate the story once at startup
if st.button("🔊 Play Story"):
    import threading
    def _play():
        import story_narration
        story_narration.narrate_story(story_narration.story_text)
    threading.Thread(target=_play, daemon=True).start()

# 2) Initialize Nova Sonic session once
if "ns" not in st.session_state:
    st.session_state.ns = NovaSonicSession()

if st.button("🚀 Start Voice Conversation"):
    # start session in background
    asyncio.get_event_loop().run_until_complete(st.session_state.ns.start())
    st.success("Nova Sonic ready—speak into mic!")

# 3) Embed real‑time mic ↔ Nova Sonic
webrtc_ctx = webrtc_streamer(
    key="nova",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_receiver_size=CHUNK_SIZE,
)

if webrtc_ctx.audio_receiver and st.session_state.ns.is_active:
    # pull all audio frames, send to Nova Sonic
    def _process_audio():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ns = st.session_state.ns
        while True:
            frame = webrtc_ctx.audio_receiver.get_frame()
            if frame is None:
                break
            pcm = frame.to_ndarray().tobytes()
            loop.run_until_complete(ns.send_audio(pcm))
            # drain Nova Sonic's response queue and play back via streamer
            while not ns.audio_queue.empty():
                chunk = loop.run_until_complete(ns.audio_queue.get())
                if chunk is None:
                    break
                webrtc_ctx.audio_sender.send_frames(frame.from_ndarray(
                    frame.to_ndarray()  # placeholder: actual frame->audio from bytes
                ))
        # pass
    threading.Thread(target=_process_audio, daemon=True).start()

st.write("Once session is ready, just speak—no further buttons needed.")
