import boto3
import pyaudio

story_text = (
    "Once upon a time in a quiet village, there lived a wise old storyteller. "
    "Every evening, children gathered to hear tales of adventure and magic. "
    "One night, the storyteller began a new tale about a hidden treasure in the forest, "
    "protected by an ancient talking tree. Little did the villagers know, this story "
    "would soon come to life, and they would become part of the adventure."
)

def narrate_story(text: str):
    """Use Amazon Polly to narrate `text` as 16 kHz PCM and play it via PyAudio."""
    polly = boto3.client('polly')
    resp = polly.synthesize_speech(
        Text=text,
        VoiceId='Matthew',
        OutputFormat='pcm',
        SampleRate='16000'   # raw PCM supports only 8000 or 16000
    )
    audio_stream = resp.get('AudioStream')
    if not audio_stream:
        raise RuntimeError("Polly returned no audio")

    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)

    chunk = 1024
    while True:
        data = audio_stream.read(chunk)
        if not data:
            break
        out.write(data)

    out.stop_stream()
    out.close()
    pa.terminate()
    audio_stream.close()

if __name__ == "__main__":
    narrate_story(story_text)
