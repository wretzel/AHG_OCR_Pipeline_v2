# server_utils/voice.py

import json
import queue
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os


class VoiceRecognizer:
    def __init__(self, model_path=None, samplerate=16000):
        # Default to resources/vosk_model relative to project root
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(
                os.path.join(base_dir, "..", "resources", "vosk_model")
            )

        print("Loading Vosk model from:", model_path)
        self.model_path = model_path
        self.samplerate = samplerate
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, samplerate)

        self.audio_q = queue.Queue()
        self.text_q = queue.Queue(maxsize=10)

        self.stream = None
        self._running = False
        self._last_partial = ""

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[audio] {status}")
        self.audio_q.put(bytes(indata))

    def start(self, device=None):
        if self._running:
            return
        self._running = True

        self.stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            device=device,   # 👈 pass the index here
            callback=self._audio_callback,
        )
        self.stream.start()

        threading.Thread(target=self._recognize_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("🛑 Voice recognition stopped")

    def _recognize_loop(self):
        while self._running:
            data = self.audio_q.get()
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").strip()
                if text:
                    if self.text_q.full():
                        try:
                            self.text_q.get_nowait()
                        except queue.Empty:
                            pass
                    self.text_q.put_nowait(text)
                self._last_partial = ""
            else:
                partial = json.loads(self.rec.PartialResult()).get("partial", "").strip()
                if partial and partial != self._last_partial:
                    print("..", partial)
                    self._last_partial = partial

    def latest_lines(self, n=1):
        items = list(self.text_q.queue)
        return items[-n:]
