from datasets import load_dataset
import pandas as pd

# 데이터셋 로드
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")

# JSON Lines 형식으로 저장
dataset.to_json("naver_news_full.jsonl", orient="records", lines=True, force_ascii=False)
