# SimLLM dataset (x_regeneration) 생성 

import pandas as pd
import openai
from tqdm import tqdm
from openai import OpenAI

# OpenAI API 키 설정
client = OpenAI(api_key="YOUR_API_KEY")  # ← API 키 입력

df = pd.read_csv("../data/kor_news/human_data/human_1000.csv")  # ← 파일 이름에 맞게 수정

# 결과 저장할 리스트
gpt_results = []

# 각 문장에 대해 GPT-3.5로 교정 수행
for text in tqdm(df['human'], desc="Proofreading with GPT-3.5"):
    try:
        prompt = f"""
        다음 문장을 의미는 그대로 유지하되, 문법적으로 더 유창하고 자연스럽게 다시 써 주세요.
        단, 문장의 말투(예: '~한다.', '~이다.' 등)는 바꾸지 말고, 원래 문장의 어미 형태를 그대로 유지하세요.
        또한, 불필요한 설명 없이 수정된 문장만 한 줄로 출력하세요.
        
        문장:
        {text}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 한국어 문장을 교정하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  
        )
        corrected_text = response.choices[0].message.content.strip()
    except Exception as e:
        corrected_text = f"[ERROR] {e}"
    
    gpt_results.append(corrected_text)

# 새로운 컬럼으로 추가
df["regen1"] = gpt_results

# 결과 저장
df.to_csv("../data/kor_news/llm_regen1_data/gpt35_regen1_1000.csv", index=False)
print("교정 완료: 'regen1_1000.csv'로 저장되었습니다.")
