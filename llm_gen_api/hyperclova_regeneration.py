# HYPERCLOVA로 TOCSIN x_regeneration 생성

import os
import uuid
import time
import random
import requests
import pandas as pd
from tqdm import tqdm

INPUT_FILE = "../data/kor_news/human_data/human_1000.csv" 
OUTPUT_FILE = "../data/kor_news/llm_regen1_data/hyperclova_regen1_1000.csv"
TEMP_FILE = "temp_backup_regen.csv"

API_URL = os.getenv("HYPERCLOVA_API_URL", "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005")
API_KEY = "YOUR_API_KEY" 

TOP_P = 0.8
TEMPERATURE = 0.3
MAX_TOKENS = 256
SLEEP_SEC = 1.0
MAX_RETRY = 5
BACKOFF_BASE = 1.5

def build_prompt(text: str) -> str:
    return f"""
    다음 문장을 의미는 그대로 유지하되, 문법적으로 더 유창하고 자연스럽게 다시 써 주세요.
    단, 문장의 말투(예: '~한다.', '~이다.' 등)는 바꾸지 말고, 원래 문장의 어미 형태를 그대로 유지하세요.
    또한, 불필요한 설명 없이 수정된 문장만 한 줄로 출력하세요.

    문장:
    {text}
    """

def call_hyperclova(prompt: str) -> str:
    request_id = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "너는 한국어 문장을 유창하게 교정/재작성하는 전문가야."},
            {"role": "user", "content": prompt}
        ],
        "topP": TOP_P,
        "temperature": TEMPERATURE,
        "maxTokens": MAX_TOKENS,
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
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text[:300]}")
            data = res.json()
            content = (
                data.get("result", {})
                    .get("message", {})
                    .get("content")
            )
            return content.strip() if content else ""
        except Exception as e:
            if attempt == MAX_RETRY:
                return f"[ERROR] {e}"
            wait = (BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(wait)

def main():
    df = pd.read_csv(INPUT_FILE)
    results = []
    for text in tqdm(df["human"], desc="Regenerating with HyperCLOVA"):
        prompt = build_prompt(text)
        regenerated = call_hyperclova(prompt)
        results.append(regenerated)
        time.sleep(SLEEP_SEC)

    df["regen1"] = results
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"regeneration 완료: '{OUTPUT_FILE}' 저장")

if __name__ == "__main__":
    if not API_KEY or API_KEY == "REPLACE_ME":
        raise SystemExit("환경변수 HYPERCLOVA_API_KEY를 설정하세요.")
    main()
