# tocsin & simllm score merge

import json
import pandas as pd
import numpy as np

output_file = './scores/indo_cross_scores_1000.csv'

# 1. TOCSIN 점수 로드 (json)
with open('../data/kor_news/result_scroe/gpt35_kobart_tocsin_score.json', 'r', encoding='utf-8') as f:
    tocsin_data = json.load(f)

tocsin_human_scores = np.array(tocsin_data['predictions']['real'])
tocsin_llm_scores = np.array(tocsin_data['predictions']['samples'])
tocsin_scores = np.concatenate([tocsin_human_scores, tocsin_llm_scores])

n_human = len(tocsin_human_scores)
n_llm = len(tocsin_llm_scores)
labels = np.concatenate([np.zeros(n_human, dtype=int), np.ones(n_llm, dtype=int)])

# 2. SimLLM 점수 로드 (csv)
simllm_df = pd.read_csv('../data/kor_news/result_scroe/gpt35_kobart_simllm_score.csv')
n_human = len(tocsin_human_scores)
n_llm = len(tocsin_llm_scores)
simllm_scores_human = simllm_df['sim_human_score'].values[:n_human]
simllm_scores_llm = simllm_df['sim_llm_score'].values[:n_llm]
simllm_scores = np.concatenate([simllm_scores_human, simllm_scores_llm])

print(f"len(tocsin_scores): {len(tocsin_scores)}")
print(f"len(simllm_scores): {len(simllm_scores)}")
print(f"len(labels): {len(labels)}")

results_df = pd.DataFrame({
    'tocsin_score': tocsin_scores,
    'simllm_score': simllm_scores,
    'label': labels,
})
results_df.to_csv(output_file, index=False)

