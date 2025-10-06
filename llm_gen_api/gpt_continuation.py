# TOCSIN dataset (x_continuation) 생성 

import json
import time
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key="YOUR_API_KEY")  # ← API 키 입력

# 파일 경로
input_file = "../data/kor_news/human_data/human_1000_prefix.json"  
output_file = "../data/kor_news/llm_cont_data/gpt35_cont_1000.json"
temp_file = "temp_backup_sampled.json"

# 데이터 불러오기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

input_data = data 
total_size = len(input_data)
generated_texts = []

# 프롬프트 템플릿
def build_prompt(prefix, target_length):
    return f"""다음 뉴스 기사 문장의 앞부분에 이어서 전체 문장을 자연스럽고 완결성 있게 작성하세요.
    전체 문장의 길이는 약 {target_length}자 내외가 되도록 해주세요. 아래는 문장의 시작 부분입니다: '{prefix.strip()}'
    """

# API 요청 루프
for i, row in enumerate(input_data):  
    try:
        original = row["original"]
        prefix = row["prefix_20tok"]
        target_length = int(len(original) * 0.95)

        prompt = build_prompt(prefix, target_length)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256
        )
        llm_generated = response.choices[0].message.content.strip()
        llm_generated = llm_generated.replace('\"', '').strip() 

        full_sentence = prefix.strip() + llm_generated
        generated_texts.append(full_sentence)


    except Exception as e:
        print(f"[{i+1}] 에러 발생: {e}")
        generated_texts.append("")

    # 진행 상황 출력
    if (i + 1) % 10 == 0 or (i + 1) == total_size:
        print(f"{i+1}/{total_size} 문장 생성 완료")

    # 50개마다 백업
    if (i + 1) % 50 == 0:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump({"sampled": generated_texts}, f, ensure_ascii=False, indent=2)
        print(f"[백업] {i+1}개 문장까지 저장됨")

    time.sleep(1.2)

# 최종 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"sampled": generated_texts}, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(generated_texts)}개 문장을 생성하여 '{output_file}'에 저장했습니다.")
