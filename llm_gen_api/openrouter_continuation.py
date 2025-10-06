# OpenRouter로 TOCSIN x_continuation 생성

import json
import time
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY",  # ← OpenRouter API 키 입력
)

input_file = "../data/kor_news/human_data/human_1000_prefix.json"  
output_file = "../data/kor_news/llm_cont_data/llama_cont_1000.json"
temp_file = "temp_backup_sampled.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "original" in data:
    originals = data.get("original", [])
    prefixes  = data.get("prefix_20tok", [])
    # prefix_20tok가 없으면 즉석 생성
    def make_prefix_20tok(text, n=20):
        toks = text.split()
        return " ".join(toks[:n])
    input_data = []
    for idx, orig in enumerate(originals):
        pref = prefixes[idx] if idx < len(prefixes) else make_prefix_20tok(orig)
        input_data.append({"original": orig, "prefix_20tok": pref})
elif isinstance(data, list):
    input_data = data
else:
    raise ValueError("입력 JSON은 리스트이거나, {'original': [...], 'prefix_20tok': [...](선택)} 형태여야 합니다.")

total_size = len(input_data)
generated_texts = []

def build_prompt(prefix, target_length):
    return (
        "다음 뉴스 기사 문장의 앞부분에 이어서 전체 문장을 자연스럽고 완결성 있게 작성하세요.\n"
        f"전체 문장의 길이는 약 {target_length}자 이상이 되도록 충분히 길게 작성하세요.\n"
        f"주어진 앞부분은 절대로 다시 반복하지 말고, 이어지는 뒷부분만 작성하세요."
        f"아래는 문장의 시작 부분입니다. 다음 문장을 다시 쓰지 말고 : '{prefix.strip()}'\n"
    )

for i, row in enumerate(input_data):
    try:
        original = row["original"]
        prefix = row["prefix_20tok"]
        target_length = int((len(original) - len(prefix)) * 0.95)

        prompt = build_prompt(prefix, target_length)

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",  # OpenRouter 모델명
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        llm_generated = response.choices[0].message.content.strip()
        llm_generated = llm_generated.replace('\"', '').strip() 

        full_sentence = prefix.strip() + llm_generated
        generated_texts.append(full_sentence)

    except Exception as e:
        print(f"[{i+1}] 에러 발생: {e}")
        generated_texts.append("")

    if (i + 1) % 10 == 0 or (i + 1) == total_size:
        print(f"{i+1}/{total_size} 문장 생성 완료")

    # 50개마다 백업
    if (i + 1) % 50 == 0:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump({"sampled": generated_texts}, f, ensure_ascii=False, indent=2)
        print(f"[백업] {i+1}개 문장까지 저장됨")

    time.sleep(4) 

with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"sampled": generated_texts}, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(generated_texts)}개 문장을 생성하여 '{output_file}'에 저장했습니다.")