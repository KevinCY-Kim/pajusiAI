# 🧠 pajusiAI – 파주시 민원 챗봇/음성AI 솔루션

## 📌 프로젝트 개요  
pajusiAI는 **파주시(支)** 행정 서비스 지원을 목표로 한 AI 기반 챗봇 및 음성 처리 시스템입니다.  
음성 인식 + 자연어 질의응답(RAG) 기능을 통해 민원인의 음성 또는 텍스트 질문을 처리하고, 빠르고 정확한 응답을 제공하는 것을 목표로 합니다.

---

## ⚙️ 주요 기능  
- 🎙️ **음성 → 텍스트 변환**: GPU 가속 기반 음성 인식(예: Whisper 기반)  
- 💬 **질의응답(RAG)**: 음성 또는 텍스트 입력을 받아 민원 문서 및 DB 등과 연동해 응답 제공  
- 🌐 **FastAPI 서버**: RESTful API 및 웹 인터페이스 제공  
- 📂 **파일 업로드 및 정적 자원 처리**: `static/` 폴더 등으로 웹 UI 및 결과물 관리  

---

## 🏗️ 저장소 구조  
pajusiAI/
│
├── static/ # 정적 파일(웹 UI, 결과 페이지 등)
├── paju_careon_fastapi.py # FastAPI 메인 엔트리포인트 (민원 서비스)
├── rag_service.py # RAG(질의응답) 관련 서비스 모듈
├── test_whisper_gpu.py # 음성 인식 및 GPU 테스트용 스크립트
└── README.md # (이 파일) 프로젝트 개요 및 정보
※ 폴더 구조는 저장소 기준이며, 이후 `app/`, `routers/`, `utils/` 등으로 리팩토링 가능.

---

## 🚀 실행 방법

### 1️⃣ 가상환경 생성  
```bash
conda create -n pajusiAI python=3.10
conda activate pajusiAI
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
