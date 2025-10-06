# OpenRouter로 TOCSIN x_regeneration 생성

import os
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY",  # ← OpenRouter API 키 입력
)

input_csv  = "../data/kor_news/human_data/human_1000.csv"      
output_csv = "../data/kor_news/llm_regen1_data/llama_regen1_1000.csv"  

MODEL = "meta-llama/llama-3.3-70b-instruct:free"   
TEMPERATURE = 0.3
SLEEP_PER_CALL = 4.0   
MAX_RETRIES = 5          

df = pd.read_csv(input_csv)

texts = df["human"].astype(str).tolist()

def build_prompt(text: str) -> str:
    return f"""
    다음 문장을 의미는 그대로 유지하되, 문법적으로 더 유창하고 자연스럽게 다시 써 주세요.
    단, 문장의 말투(예: '~한다.', '~이다.' 등)는 바꾸지 말고, 원래 문장의 어미 형태를 그대로 유지하세요.
    또한, 불필요한 설명 없이 수정된 문장만 한 줄로 출력하세요.

    문장:
    {text}
    """.strip()

def call_openrouter(prompt: str):
    """429 등 오류에 대비한 재시도 포함 호출"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "당신은 한국어 문장을 교정하는 전문가입니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Rate limit" in msg or "temporarily" in msg.lower():
                wait = min(10 * attempt, 60)
                print(f"[재시도 {attempt}/{MAX_RETRIES}] Rate limit/일시 에러 감지 → {wait}s 대기")
                time.sleep(wait)
                continue
            return f"[ERROR] {msg}"
    return "[ERROR] Max retries exceeded"

results = []
for i, text in enumerate(tqdm(texts, desc="Regeneration with OpenRouter")):
    prompt = build_prompt(text)
    out = call_openrouter(prompt)
    results.append(out)
    time.sleep(SLEEP_PER_CALL)

    if (i + 1) % 20 == 0:
        df["regen1"] = results + [""] * (len(df) - len(results))
        df.to_csv("temp_backup.csv", index=False)
        print(f"백업 저장 완료: {i+1}문장까지 temp_backup.csv 저장됨")

df["regen1"] = results
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"재작성 완료: '{output_csv}'로 저장했습니다.")
