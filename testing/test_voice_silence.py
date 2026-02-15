import time
from voice.subtitle_engine import SubtitleEngine

s = SubtitleEngine(silence_threshold=0.5, min_output_interval=0.0)
# simulate growing partials
s.process_partial('he')
# ignored (too small)
s.process_partial('hello')
print('partial_buffer after hello:', s.partial_buffer)
# simulate silence ticks
for i in range(6):
    s.process_partial('')
    time.sleep(0.1)

print('queue contents:', list(s.text_q.queue))
