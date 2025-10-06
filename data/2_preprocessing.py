import json
import re

# 입력 JSONL 파일 경로
input_path = "naver_news_full.jsonl"
# 출력 JSON 파일 경로
output_path = "naver_news_preprocessed.json"

summaries = set()  # set 사용 → 중복 자동 제거

# 특수기호 정규식 (문장부호 제외)
special_char_pattern = re.compile(r"[■◆●★▲▶◀♣♠♥♦※▷◁→←↑↓…☆★◇◆○◎●□■△▲▽▼◈▣▤▥▦▧▨▩◐◑♤♧♣⊙⌒∴∑㉿ⓒⓢ™®]")

# 한자 정규식 추가
hanja_pattern = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        summary = data.get("summary", "")
        
        # 전처리
        summary = summary.strip()  # 앞뒤 공백 제거
        summary = special_char_pattern.sub("", summary)  # 특수기호 제거
        summary = hanja_pattern.sub("", summary)         # 한자 제거
        
        if summary:  # 빈 문자열/None 제외
            summaries.add(summary)

# 리스트로 변환 후 JSON 저장
output_data = {"original": list(summaries)}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)



print(f"{len(summaries)}개의 summary가 'original' 리스트로 저장되었습니다.")
