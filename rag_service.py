# rag_service.py
# 파주시 전용 RAG MVP (키워드 기반 간단 검색 버전)
from typing import List, Dict

# 실제 환경에선 여기에 파주시 공식 문서들을 전처리하여 넣거나,
# 외부 JSON/DB/벡터DB에서 로드하도록 교체하면 됨.
# 지금은 동작 확인용 더미 데이터.
PAJU_DOCS: List[Dict[str, str]] = [
    {
        "title": "파주시청 대표 연락처",
        "content": "파주시청 대표번호는 031-940-2114입니다. 일반 민원 문의는 이 번호로 연락할 수 있습니다."
    },
    {
        "title": "노인 복지 서비스",
        "content": "만 65세 이상 어르신을 위한 기초연금, 경로당 운영, 방문 건강 관리 서비스 등은 파주시 복지정책과 또는 행정복지센터에서 안내받을 수 있습니다."
    },
    {
        "title": "장애인 복지 지원",
        "content": "장애인 등록, 활동 지원, 이동 지원, 보조기기 지원 등 문의는 파주시 장애인복지과로 문의하세요."
    },
    {
        "title": "보건소 안내",
        "content": "파주시 보건소에서는 예방접종, 건강검진, 만성질환 관리, 치매안심센터 서비스를 제공합니다."
    },
]

def _score(query: str, text: str) -> int:
    """아주 단순한 키워드 매칭 점수 (MVP용)"""
    q_words = [w for w in query.lower().split() if len(w) > 1]
    t_lower = text.lower()
    return sum(1 for w in q_words if w in t_lower)

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    질문과 관련 있어 보이는 문서 상위 top_k를 이어붙여 context로 반환.
    RAG 고도화 시 이 부분만 교체하면 됨.
    """
    scored = []
    for doc in PAJU_DOCS:
        s = _score(query, doc["content"])
        if s > 0:
            scored.append((s, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return ""

    selected = [doc["title"] + "\n" + doc["content"] for _, doc in scored[:top_k]]
    return "\n\n".join(selected)
