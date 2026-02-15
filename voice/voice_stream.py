# voice/voice_stream.py

from socket import timeout
import sounddevice as sd
import queue


class VoiceStream:
    """
    Microphone audio stream abstraction.
    Handles:
      - opening microphone
      - audio callback
      - pushing audio bytes into a queue
      - clean start/stop lifecycle
    """

    def __init__(self, samplerate=16000, blocksize=4000, channels=1, device=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.device = device

        self.audio_q = queue.Queue()
        self.stream = None

    # ------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        self.audio_q.put(bytes(indata))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def start(self):
        """
        Start microphone capture.
        """
        if self.stream:
            return

        self.stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=self.channels,
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        """
        Stop microphone capture.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio(self, timeout=0.1):
        try:
            return self.audio_q.get(timeout=timeout)
        except queue.Empty:
            # No audio available right now — return None to signal silence
            return None

