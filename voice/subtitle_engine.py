# voice/subtitle_engine.py
import time
import queue
from .punctuation import infer_punctuation


class SubtitleEngine:
    """
    Handles:
      - partial buffering (stable only)
      - silence detection
      - sentence boundaries
      - smoothing + punctuation
      - output queue for subtitles
    """

    def __init__(self, silence_threshold=1.0, min_output_interval=0.4):
        self.silence_threshold = silence_threshold
        self.min_output_interval = min_output_interval
        # How long a committed subtitle remains visible (seconds)
        self.subtitle_timeout = 2.0

        self.last_voice_time = time.time()
        self.last_output_time = 0

        self.partial_buffer = ""
        self.last_partial = ""
        self.current_subtitle = ""

        self.text_q = queue.Queue(maxsize=10)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def process_partial(self, partial: str):
        """
        Handle partial text from recognizer.
        Partial rules:
          - ignore empty partials
          - ignore shrinking partials
          - ignore tiny/noisy partials (< 3 chars)
          - only accept if it grows meaningfully
        """
        now = time.time()

        if partial:
            self.last_voice_time = now

            # Ignore tiny/noisy partials
            if len(partial) < 3:
                return

            # Ignore shrinking partials (Vosk sometimes backtracks)
            if len(partial) < len(self.last_partial):
                return

            # Ignore partials that barely change
            if abs(len(partial) - len(self.last_partial)) < 2:
                return

            # Accept stable partial
            self.partial_buffer = partial
            self.last_partial = partial
            return

        # No partial text → check for silence
        if now - self.last_voice_time > self.silence_threshold:
            # If we have a stable partial buffered, promote it to current
            # so it can be committed as a subtitle on silence.
            if self.partial_buffer:
                self.current_subtitle = self._smooth(self.partial_buffer)
                self.partial_buffer = ""
                self.last_partial = ""

            self._commit_subtitle()

    def process_final(self, final_text: str):
        """
        Handle final text from recognizer.
        Final rules:
          - final replaces partial (never merged)
          - clear partial buffer
          - smooth + punctuate
        """
        if not final_text:
            return

        # Final replaces partials entirely
        cleaned = self._smooth(final_text)

        self.current_subtitle = cleaned
        self.partial_buffer = ""
        self.last_partial = ""

        self._commit_subtitle()

    def latest_lines(self, n=1):
        # Drop expired subtitle(s) if last output is older than timeout
        now = time.time()
        if self.last_output_time and (now - self.last_output_time > self.subtitle_timeout):
            # clear the queue
            try:
                while True:
                    self.text_q.get_nowait()
            except queue.Empty:
                pass
            # reset last_output_time so we don't repeatedly clear
            self.last_output_time = 0
            return []

        items = list(self.text_q.queue)
        return items[-n:]

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _commit_subtitle(self):
        """
        Push the current subtitle to the queue if enough time has passed.
        """
        if not self.current_subtitle:
            return

        now = time.time()
        if now - self.last_output_time < self.min_output_interval:
            return

        if self.text_q.full():
            try:
                self.text_q.get_nowait()
            except queue.Empty:
                pass

        self.text_q.put_nowait(self.current_subtitle)
        self.last_output_time = now
        self.current_subtitle = ""

    def _smooth(self, text: str) -> str:
        """
        Clean up text:
          - trim
          - collapse spaces
          - capitalize
          - add punctuation
        """
        text = " ".join(text.strip().split())
        if not text:
            return ""

        # Capitalize
        text = text[0].upper() + text[1:]

        # Add punctuation
        text = infer_punctuation(text)

        return text
