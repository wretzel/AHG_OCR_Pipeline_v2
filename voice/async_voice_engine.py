# voice/async_voice_engine.py

import threading
import time

from voice.recognizer import VoskRecognizer
from voice.subtitle_engine import SubtitleEngine
from voice.voice_stream import VoiceStream


class AsyncVoiceEngine:
    """
    High-level async voice engine.
    Handles:
      - microphone stream
      - background recognition thread
      - routing partial/final text
      - subtitle engine updates
      - exposing latest subtitle
    """

    def __init__(
        self,
        model_path,
        samplerate=16000,
        silence_threshold=1.0,
        min_output_interval=0.4,
    ):
        self.model_path = model_path
        self.samplerate = samplerate

        # Core components
        self.recognizer = VoskRecognizer(model_path, samplerate=samplerate)
        self.subtitles = SubtitleEngine(
            silence_threshold=silence_threshold,
            min_output_interval=min_output_interval,
        )
        self.stream = VoiceStream(samplerate=samplerate)

        # Thread control
        self._running = False
        self._thread = None

    # ------------------------------------------------------------
    # Background recognition loop
    # ------------------------------------------------------------

    def _recognition_loop(self):
        # Emit periodic empty partials while silent so SubtitleEngine
        # can detect silence and clear subtitles after silence_threshold.
        silence_tick_interval = 0.2  # seconds between empty-partial ticks
        last_silence_tick = 0

        while self._running:
            data = self.stream.get_audio(timeout=0.1)
            now = time.time()

            # No audio available -> emit empty partial periodically
            if not data:
                if now - last_silence_tick >= silence_tick_interval:
                    last_silence_tick = now
                    self.subtitles.process_partial("")
                continue

            # We received audio, reset silence tick clock
            last_silence_tick = now

            if self.recognizer.accept_audio(data):
                final_text = self.recognizer.get_final()
                self.subtitles.process_final(final_text)
            else:
                partial = self.recognizer.get_partial()
                self.subtitles.process_partial(partial)


    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def start(self):
        """
        Start microphone + recognition thread.
        """
        if self._running:
            return

        self._running = True
        self.stream.start()

        self._thread = threading.Thread(
            target=self._recognition_loop, daemon=True
        )
        self._thread.start()

    def stop(self):
        """
        Stop microphone + recognition thread.
        """
        self._running = False
        self.stream.stop()

        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    def get_latest(self):
        """
        Return the latest committed subtitle line (or empty string).
        """
        lines = self.subtitles.latest_lines(1)
        return lines[0] if lines else ""
    
    def tick(self):
        # Force subtitle engine to update timers
        self.subtitles.latest_lines(1)

