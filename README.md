# 파주시 AI 민원 상담 챗봇 '돌봄온' (MVP)

이 프로젝트는 파주시청의 민원, 복지, 건강 관련 문의에 답변하는 AI 음성 챗봇 '돌봄온'의 MVP(Minimum Viable Product) 버전입니다.

FastAPI를 기반으로 STT(Speech-to-Text), RAG(Retrieval-Augmented Generation), LLM(Large Language Model), TTS(Text-to-Speech) 기술을 통합하여 사용자에게 음성 기반의 대화형 서비스를 제공합니다.

##  시연 영상

[!돌봄온 시연 영상](https://youtube.com/shorts/gLKFxrx5eSY)
> (이미지를 클릭하면 유튜브 시연 영상으로 이동합니다.)

## ✨ 주요 기능

- **음성 대화**: 마이크를 통해 질문하고 음성으로 답변을 받습니다.
- **고성능 STT**: `faster-whisper`를 GPU(CUDA) 환경에서 실행하여 빠르고 정확한 음성 인식을 수행합니다.
- **문서 기반 답변 (RAG)**: `rank-bm25`를 사용하여 파주시 조례, 규정, 연락처 등 내부 데이터베이스에서 질문과 가장 관련 높은 문서를 검색하고, 이를 LLM의 답변 생성에 활용합니다.
- **LLM 답변 생성**: SKT의 A.X(에이닷엑스) GPT 모델을 활용하여 검색된 정보를 바탕으로 자연스러운 답변을 생성합니다.
- **TTS 음성 출력**: `gTTS`를 사용하여 생성된 텍스트 답변을 음성 파일로 변환하여 제공합니다.

## ⚙️ 시스템 아키텍처

```
┌───────────────┐      ┌───────────────────┐      ┌───────────────────┐
│   User        │----->│ FastAPI Server    │----->│   RAG Service     │
│(Browser/Mic)  │      │(paju_careon_fastapi.py) │      │   (rag_service.py)  │
└───────────────┘      └───────────────────┘      └───────────────────┘
       ▲                 | 1. Audio Upload          | 4. Retrieve Context
       |                 | 2. STT (faster-whisper)  |
       |                 | 3. Query               |
       |                 | 5. Generate Answer     |
       |                 | 6. TTS (gTTS)            |
       |                 | 7. Return JSON         |
       |                 └────────────────────┬───┘
       |                                      |
       └──────────────────────────────────────┘
                         (Text, TTS URL)        |
                                                v
                                      ┌───────────────────┐
                                      │    SKT GPT API    │
                                      └───────────────────┘
```
2️⃣ 의존성 설치
pip install -r requirements.txt


requirements.txt 파일이 없다면 주요 라이브러리(fastapi, uvicorn, whisper, transformers 등)를 직접 명시하고 설치하세요.

3️⃣ 서버 실행
uvicorn paju_careon_fastapi:app --host 0.0.0.0 --port 8000 --reload


웹 브라우저에서 http://localhost:8000 접속하면 서비스 인터페이스 또는 API 문서를 확인할 수 있습니다.

🧠 API 예시
✅ 음성 인식 요청 (POST)
POST /stt
Content-Type: multipart/form-data
file: <음성파일(.wav/.mp3)>


응답 예시

{
  "text": "안녕하세요, 파주시 인공지능 챗봇입니다."
}

✅ 질의응답 요청 (POST)
POST /rag
Content-Type: application/json
{
  "question": "주차요금 할인 기준 알려줘"
}


응답 예시

{
  "answer": "파주시 주차요금 할인 기준은 …"
}


실제 엔드포인트 이름 및 파라미터는 rag_service.py 코드를 참고해 주세요.

🧩 개발 환경 및 기술 스택

Python 3.10

FastAPI

Uvicorn

Whisper 또는 HuggingFace Transformers

CUDA-enabled GPU (NVIDIA)

GitHub 저장소 기반 버전관리

🧱 향후 개선 방향

대용량 민원 문서용 인덱싱 및 Vector DB 연동 (예: Pinecone, Milvus)

음성 명령 기반 행정 서비스 자동화 (예: “민원 접수해줘”)

UI 고도화 및 챗봇 대시보드 구현

Whisper Large-v3 등 최신 음성모델 적용 → 인식률 향상

사내 내부용/외부용 챗봇 분리 운영 구조 설계 (보안형 vs 공개형)

👤 제작자

KevinCY Kim (김창용)

GitHub: https://github.com/KevinCY-Kim

이메일: stonez788@gmail.com

⚡ 이 프로젝트는 파주시 공공서비스 AI 챗봇 PoC 목적으로 개발되었습니다.
