# testing/test_modules/voice_tester.py
import time
from server_utils.voice import VoiceRecognizer

def main():
    vr = VoiceRecognizer(model_path="resources/vosk_model", samplerate=16000)
    vr.start()

    print("Speak into the mic. Printing recognized lines...\n")
    try:
        while True:
            lines = vr.latest_lines(n=3)
            if lines:
                print(">>", " | ".join(lines))
            time.sleep(0.5)
    except KeyboardInterrupt:
        vr.stop()

if __name__ == "__main__":
    main()
