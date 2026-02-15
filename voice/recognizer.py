# voice/recognizer.py
import json
from vosk import Model, KaldiRecognizer


class VoskRecognizer:
    """
    Thin wrapper around Vosk ASR.
    Handles:
      - model loading
      - feeding audio chunks
      - retrieving partial/final text
    """

    def __init__(self, model_path, samplerate=16000):
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, samplerate)

    def accept_audio(self, data: bytes) -> bool:
        """
        Feed audio bytes to Vosk.
        Returns True if a final result is ready.
        """
        return self.rec.AcceptWaveform(data)

    def get_partial(self) -> str:
        """
        Return partial text (unstable).
        """
        try:
            return json.loads(self.rec.PartialResult()).get("partial", "").strip()
        except Exception:
            return ""

    def get_final(self) -> str:
        """
        Return final text (stable).
        """
        try:
            return json.loads(self.rec.Result()).get("text", "").strip()
        except Exception:
            return ""
