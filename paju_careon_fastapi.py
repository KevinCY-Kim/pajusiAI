# paju_careon_fastapi.py
# FastAPI + STT + RAG + GPT + TTS + 마이크 버튼 HTML (MVP)

import os
# os.environ["CT2_USE_CUDNN"] = "0"

import re 
import time
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from gtts import gTTS

from rag_service import retrieve_context
from rag_service import clean_llm_output # LLM이 생성 후 클린작업

# ─────────────────────────────
# 1. 환경 설정 및 로깅
# ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paju_fastapi")

load_dotenv()

SKT_API_KEY = os.getenv("ADOTX_API_KEY")
if not SKT_API_KEY:
    raise RuntimeError("ADOTX_API_KEY 환경변수가 설정되어 있지 않습니다.")

# ─────────────────────────────
# 2. SKT GPT 클라이언트 설정
# ─────────────────────────────
GPT_CLIENT = OpenAI(
    api_key=SKT_API_KEY,
    base_url="https://guest-api.sktax.chat/v1",
)

MODEL = "ax4"

# ─────────────────────────────
# 3. STT(Whisper) 모델 로드
# ─────────────────────────────
# CPU에서도 돌아가도록 경량 설정(MVP). GPU 있으면 device="cuda"로 교체 가능.
# GPU 있는 서버 기준으로 변환 (음성인식 퀄리티 높이기 위함)
# STT_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
STT_MODEL = WhisperModel("medium", device="cuda", compute_type="float16")


# ─────────────────────────────
# 4. 시스템 프롬프트
# ─────────────────────────────
SYSTEM_PROMPT = """당신은 파주시청에서 운영하는 AI 민원·복지·건강 안내 상담사 '돌봄온'입니다.
시민이 이해하기 쉬운 말로, 파주시 관련 행정, 복지, 건강, 조례 정보를 제공합니다.

역할과 원칙:
- 파주시 관련 공식 문서(조례, 지침 등)를 근거로 답변하는 것을 우선합니다.
- 답변은 2~3줄 정도의 짧은 문단으로, 핵심만 명확하게 설명합니다.
- 관련 부서, 전화번호, 제도명이 있을 경우 구체적으로 명시합니다.
- 모를 경우 임의로 추측하지 말고
  "해당 정보는 파주시청 민원콜센터(031-940-2114)로 문의하세요."라고 안내합니다.
- 질문이 조례의 특정 조문(예: 제1조, 제2조 등)에 대한 것이라면,
  마지막 줄에 (○○조례 제n조 제목) 형태로 간단히 근거를 표기합니다.
- LLM이 답변을 생성한 후에 예를 들어 **파주시 보건소**이렇게 출력될 경우 앞뒤에 표기된 **는 제외하고 출력하세요.
- LLM이 답변을 생성한 후에 출력에서 ###이 있는 경우 제외하고 출력하세요.
- ()이런 괄호는 읽지 않습니다.
- **, ### 이런 기호는 읽지 않습니다.
"""

# 정규표현식
def strip_parentheses_for_tts(text: str) -> str:
    """
    TTS용 텍스트에서 () 괄호 안의 내용을 제거.
    예: '... 안내합니다. (파주시 조례 제1조 목적)' -> '... 안내합니다.'
    """
    # 1. () 안의 모든 내용 제거
    no_paren = re.sub(r'\([^)]*\)', '', text)
    # 2. 중복 공백 정리
    cleaned = " ".join(no_paren.split())
    return cleaned

# ─────────────────────────────
# 5. STT
# ─────────────────────────────
def speech_to_text(audio_path: str) -> str:
    try:
        segments, _ = STT_MODEL.transcribe(
            audio_path,
            beam_size=5,
            language="ko",            # 🔴 한국어 고정 (음성인식 퀄리티 올리기 위함)
            vad_filter=True,          # 🔴 무음/잡음 컷
            temperature=0,            # 🔴 온도 낮게 (결과 안정화)
            vad_parameters={"min_silence_duration_ms": 500}
        )
        text = " ".join([seg.text for seg in segments]).strip()
        logger.info(f"[STT 결과]: {text}")
        return text
    except Exception as e:
        logger.error(f"STT 변환 실패: {e}")
        return ""

# ─────────────────────────────
# 6. RAG + GPT
# ─────────────────────────────
def generate_answer_with_rag(query: str) -> str:
    """
    1) query로 문서 검색 (RAG, top1)
    2) context와 함께 GPT 호출 (2~3줄 요약)
    """
    try:
        context = retrieve_context(query)

        user_prompt = f"""
다음은 파주시 관련 공식 문서(조례 등)에서 검색된 내용입니다. 필요할 때만 참고하세요.

[관련 조례/문서 내용]
{context if context else "검색된 문서가 없습니다. 문서가 없을 경우, 일반적인 상식 선에서만 안내하세요."}

[시민 질문]
{query}

[요청]
1. 위 '관련 조례/문서 내용' 중에서 질문과 직접적으로 관련된 핵심 내용만 골라 주세요.
2. 시민이 이해하기 쉬운 자연스러운 한국어로, 2~3줄 정도의 짧은 문단으로 요약해서 답변하세요.
3. 숫자, 연령, 금액, 조문 번호 등은 최대한 원문과 동일하게 유지하세요.
4. 질문이 특정 조례의 조문(예: 제1조, 제2조 등)에 대한 것이라면,
   마지막 줄에 (○○조례 제n조 제목)과 같이 간단히 근거를 괄호 안에 표기하세요.
5. 문서 내용이 없거나 불충분하다고 판단되면, 임의로 추측하지 말고
   "정확한 안내는 파주시청 민원콜센터(031-940-2114)에 문의해 주세요."라는 문장을 마지막에 덧붙이세요.
6. "OO팀 연락처"를 묻는 경우  
   해당 팀 소속 직원들의 전화번호 전체를 제공한다.  
7. 데이터에 팀명이 여러개 있으면 "duties" 를 열거해주고, 다음질문에 duties와 팀을 맵핑하여 답변할 수 있도록, 명시적으로 다시 질문하도록 답변한다.
"""

        completion = GPT_CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content.strip()
        logger.info(f"[돌봄온 RAG 응답]: {answer}")
        return answer
    except Exception as e:
        logger.error(f"GPT(RAG) 응답 실패: {e}")
        return "현재 상담 시스템에 문제가 발생했습니다. 잠시 후 다시 이용해 주세요."

# ─────────────────────────────
# 7. TTS
# ─────────────────────────────
def text_to_speech(text: str, out_path: str) -> str:
    try:
        tts = gTTS(text=text, lang="ko")
        tts.save(out_path)
        logger.info(f"[TTS 생성 완료]: {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"TTS 변환 실패: {e}")
        return ""

# ─────────────────────────────
# 8. FastAPI 앱 및 정적 파일
# ─────────────────────────────
app = FastAPI(
    title="파주시 돌봄온 챗봇 API (MVP)",
    description="STT + RAG + GPT + TTS + 마이크 버튼 기반 음성 챗봇",
    version="0.1.0",
)

# static/tts 디렉토리 준비 (음성 파일 서빙용)
os.makedirs("static/tts", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────
# 9. 웹 UI (마이크 버튼 페이지)
# ─────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """테스트용 웹 UI"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>index.html이 없습니다. static/index.html 을 생성해 주세요.</h1>",
            status_code=500,
        )

# ─────────────────────────────
# 10. 음성 기반 대화 엔드포인트
# ─────────────────────────────
@app.post("/paju/voice-chat")
async def voice_chat(
    user_id: str = Form("guest"),
    audio: UploadFile | None = None,
):
    if not audio:
        return JSONResponse({"error": "음성 파일이 필요합니다."}, status_code=400)

    # 1) 음성 파일 임시 저장
    ts = int(time.time())
    tmp_input = f"/tmp/{user_id}_{ts}_{audio.filename}"
    with open(tmp_input, "wb") as f:
        f.write(await audio.read())

    # 2) STT
    query = speech_to_text(tmp_input)
    if not query:
        return JSONResponse({"error": "음성 인식 실패"}, status_code=500)

    # 3) RAG + GPT
    answer = generate_answer_with_rag(query)
    answer = clean_llm_output(answer) 
    

    # 4) TTS -> static/tts 경로에 저장 (브라우저에서 바로 접근 가능)
    tts_filename = f"{user_id}_{ts}.mp3"
    tts_path = os.path.join("static", "tts", tts_filename)
    saved_path = text_to_speech(answer, out_path=tts_path)
    if not saved_path:
        return JSONResponse({"error": "TTS 변환 실패"}, status_code=500)

    tts_url = f"/static/tts/{tts_filename}"

    # 5) 결과 반환 (프론트에서 채팅+음성 재생)
    return JSONResponse(
        {
            "user_id": user_id,
            "recognized_text": query,
            "answer": answer,
            "tts_url": tts_url,
        }
    )

# ─────────────────────────────
# 11. Uvicorn 직접 실행
# ─────────────────────────────
# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run("paju_careon_fastapi:app", host="0.0.0.0", port=8903, reload=True)

# 실행명령 : python -m uvicorn paju_careon_fastapi:app --reload --host 0.0.0.0 --port 8900