import json
import random

# 입력 파일 경로
input_path = "naver_news_preprocessed.json"
# 출력 파일 경로
output_path = "human_1000.json"

# 파일 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_summaries = data.get("original", [])

# 샘플 수 설정
sample_size = 1000
if len(all_summaries) < sample_size:
    raise ValueError(f"데이터가 {sample_size}개보다 적습니다. 현재 {len(all_summaries)}개.")

# 랜덤 샘플링
sampled_summaries = random.sample(all_summaries, sample_size)

# 저장 형식 유지
output_data = {"original": sampled_summaries}

# JSON 파일로 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"{sample_size}개의 문장을 랜덤 추출하여 '{output_path}'에 저장했습니다.")
