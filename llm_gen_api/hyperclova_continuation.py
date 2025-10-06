# HYPERCLOVA로 TOCSIN x_continuation 생성

import os
import json
import time
import uuid
import random
import requests
import re

INPUT_FILE = "../data/kor_news_data/human_data/human_1000_prefix.json"
OUTPUT_FILE = "../data/kor_news_data/llm_cont_data/hyperclova_cont_1000.json"
TEMP_FILE = "temp_backup_cont.json"

API_URL = os.getenv("HYPERCLOVA_API_URL", "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005")
API_KEY = "YOUR_API_KEY" 

TOP_P = 0.8
TEMPERATURE = 0.3
MAX_TOKENS = 256
SLEEP_SEC = 1.2
MAX_RETRY = 5
BACKOFF_BASE = 1.5 

def build_prompt(prefix: str, target_length_chars: int) -> str:
    return (
        "다음 뉴스 기사 문장의 앞부분에 이어서 전체 문장을 자연스럽고 완결성 있게 작성하세요.\n"
        f"전체 문장의 길이는 약 {target_length_chars}자 내외가 되도록 하세요.\n"
        f"주어진 앞부분은 절대로 다시 반복하지 말고, 이어지는 뒷부분만 작성하세요."
        f"아래는 문장의 시작 부분입니다. 다음 문장을 다시 쓰지 말고 : '{prefix.strip()}'\n"
    )

def call_hyperclova(prompt: str) -> str:
    """HyperCLOVA Chat Completions 호출. content 문자열을 반환."""
    request_id = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "너는 뉴스 기사 문장을 자연스럽게 완성하는 한국어 작문 어시스턴트야."},
            {"role": "user", "content": prompt}
        ],
        "topP": TOP_P,
        "temperature": TEMPERATURE,
        "maxTokens": MAX_TOKENS
    }

    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if res.status_code // 100 != 2:
                if res.status_code in (429, 500, 502, 503, 504):
                    wait = (BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(wait)
                    continue
                else:
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text[:500]}")

            data = res.json()
            content = (
                data.get("result", {})
                    .get("message", {})
                    .get("content")
            )
            if not content:
                raise ValueError(f"응답에서 content를 찾지 못했습니다: {str(data)[:500]}")
            return content.strip()
        except Exception as e:
            if attempt == MAX_RETRY:
                print(f"[WARN] HyperCLOVA 요청 실패 (시도 {attempt}/{MAX_RETRY}): {e}")
                return ""
            wait = (BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(wait)


def concat_without_dup(prefix: str, completion: str) -> str:
    p = prefix.strip()
    c = completion.strip()

    max_ol = 0
    max_len = min(len(p), len(c))
    for L in range(max_len, 0, -1):
        if p[-L:] == c[:L]:
            max_ol = L
            break
    joined = p + c[max_ol:]

    joined = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', joined)            
    joined = re.sub(r'([가-힣]{4,20})\1+', r'\1', joined)            
    return joined



def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    total_size = len(input_data)
    print(f"[INFO] 입력 샘플 수: {total_size}")

    generated_texts = []

    for i, row in enumerate(input_data, start=1):
        original = row.get("original", "")
        prefix = row.get("prefix_20tok", "")

        target_len = int(len(original) * 0.95) if original else 120

        prompt = build_prompt(prefix, target_len)
        completion = call_hyperclova(prompt)

        completion = completion.replace('"', '').strip()

        full_sentence = concat_without_dup(prefix, completion)
        generated_texts.append(full_sentence)

        if (i % 10 == 0) or (i == total_size):
            print(f"[INFO] {i}/{total_size} 문장 생성 완료")

        # 주기적 백업
        if i % 50 == 0:
            with open(TEMP_FILE, "w", encoding="utf-8") as tf:
                json.dump({"sampled": generated_texts}, tf, ensure_ascii=False, indent=2)
            print(f"[BACKUP] {i}개 문장까지 임시 저장됨 -> {TEMP_FILE}")

        time.sleep(SLEEP_SEC)

    # 최종 저장
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"sampled": generated_texts}, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(generated_texts)}개 문장을 생성하여 '{OUTPUT_FILE}'에 저장했습니다.")

if __name__ == "__main__":
    if API_KEY == "REPLACE_ME" or not API_KEY:
        raise SystemExit("환경변수 HYPERCLOVA_API_KEY를 설정하세요.")
    main()
