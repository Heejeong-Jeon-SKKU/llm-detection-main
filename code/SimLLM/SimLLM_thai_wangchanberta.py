import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class WangchanBERTaSim:
    def __init__(self, device=None, checkpoint="airesearch/wangchanberta-base-att-spm-uncased", max_length=512):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).eval()

    @torch.no_grad()
    def _embed(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        out = self.model(**enc)  # last_hidden_state: (B, L, H)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)

        # mean pooling (마스크 적용)
        summed = (hidden * mask).sum(dim=1)                  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-6)             # (B, 1)
        emb = summed / counts                                # (B, H)
        emb = F.normalize(emb, p=2, dim=-1)                  # 정규화
        return emb

    @torch.no_grad()
    def score(self, texts_a, texts_b, batch_size=32):
        assert len(texts_a) == len(texts_b), "texts_a와 texts_b 길이가 같아야 합니다."
        sims = []
        for i in range(0, len(texts_a), batch_size):
            a_batch = texts_a[i:i+batch_size]
            b_batch = texts_b[i:i+batch_size]
            ea = self._embed(a_batch)    # (B, H)
            eb = self._embed(b_batch)    # (B, H)
            cos = (ea * eb).sum(dim=-1)  # (B,)
            sims.extend(cos.detach().cpu().tolist())
        return sims


def main():
    # 경로 설정
    input_file  = "/data/jupyter/heejeong/llm-generated_text_detection_v2/data/thai_data/llm_regen2_data/thaigov_regen2_1000.csv"
    output_file = "/data/jupyter/heejeong/llm-generated_text_detection_v2/data/thai_data/result_score/thai_wangchanberta_simllm_score.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # 데이터 로드 (컬럼명 확인: human / regen1 / regen2)
    df = pd.read_csv(input_file)
    for col in ["human", "regen1", "regen2"]:
        if col not in df.columns:
            raise ValueError(f"입력 CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    df["human"]  = df["human"].astype(str).fillna("")
    df["regen1"] = df["regen1"].astype(str).fillna("")
    df["regen2"] = df["regen2"].astype(str).fillna("")

    # 스코어러 초기화
    scorer = WangchanBERTaSim(device=device, checkpoint="airesearch/wangchanberta-base-att-spm-uncased", max_length=512)

    # 유사도 계산
    print("🔍 Calculating LLM ↔ LLM similarity (regen1 vs regen2)...")
    sim_gpt_gpt = scorer.score(df["regen1"].tolist(), df["regen2"].tolist(), batch_size=32)

    print("🔍 Calculating Human ↔ LLM similarity (human vs regen1)...")
    sim_human_gpt = scorer.score(df["human"].tolist(), df["regen1"].tolist(), batch_size=32)

    # 저장
    df["sim_llm_score"]   = sim_gpt_gpt
    df["sim_human_score"] = sim_human_gpt
    df.to_csv(output_file, index=False)
    print(f"✅ Saved with both similarity scores to: {output_file}")

    # (선택) ROC-AUC 평가 예시
    try:
        from sklearn.metrics import roc_auc_score
        scores = sim_gpt_gpt + sim_human_gpt
        labels = [1] * len(sim_gpt_gpt) + [0] * len(sim_human_gpt)
        roc_auc = roc_auc_score(labels, scores)
        print(f"📈 ROC AUC score: {roc_auc:.4f}")
    except Exception as e:
        print(f"[WARN] ROC AUC 계산 생략: {e}")


if __name__ == "__main__":
    main()
