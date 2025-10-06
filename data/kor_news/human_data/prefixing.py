### human text앞의 20, 15토큰만 추출하여 저장하는 코드

import json
from transformers import AutoTokenizer

input_file = 'human_200.json'
output_file = 'human_200_prefix.json'

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_texts = data['original']
processed = []

for original in original_texts:
    tokens = tokenizer.tokenize(original)
    prefix_15 = tokenizer.convert_tokens_to_string(tokens[:15])
    prefix_20 = tokenizer.convert_tokens_to_string(tokens[:20])

    processed.append({
        "original": original,
        "prefix_15tok": prefix_15,
        "prefix_20tok": prefix_20
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"✅ {len(processed)}개 문장에 대해 prefix 추출 완료 및 '{output_file}' 저장 완료")
