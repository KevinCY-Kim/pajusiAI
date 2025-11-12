import os
os.environ["CT2_USE_CUDNN"] = "0"  # ✅ 맨 위. 어떤 import보다 먼저.

from faster_whisper import WhisperModel
import time

AUDIO_PATH = "/home/alpaco/homework/kimcy/static/tts/guest_1762906509.mp3"  # 실제 파일

start = time.time()
model = WhisperModel("small", device="cuda", compute_type="float16")
print("✅ 모델 로드 완료 (cuda)")
print("로드 시간:", round(time.time() - start, 2), "초")

start = time.time()
segments, info = model.transcribe(
    AUDIO_PATH,
    language="ko",
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 400}
)
print("STT 시간:", round(time.time() - start, 2), "초")

for seg in segments:
    print(seg.text)