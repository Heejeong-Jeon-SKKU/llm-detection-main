# sim_human_score, sim_llm_score 계산하는 코드
# kobart(기본) 사용 

import os
import pandas as pd
import torch
import torch.nn as nn
# from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- (선택) BARTpho-word 전처리 훅: VnCoreNLP로 단어 분절 + 성조정규화 후 'Chúng_tôi' 형식으로 합치기 ---
def preprocess_vi_word(texts):
    # TODO: VnCoreNLP를 붙이세요. (예시)
    # from vncorenlp import VnCoreNLP
    # rdr = VnCoreNLP(...); return [' '.join(['_'.join(sent) for sent in rdr.tokenize(t)]) for t in texts]
    return texts  # syllable 버전이면 그대로 통과

class ViBARTScorer:
    # def __init__(self, device='cuda:0', max_length=1024, checkpoint='vinai/bartpho-syllable', use_word=False):
    #     self.device = device
    #     self.max_length = max_length
    #     self.use_word = use_word
    #     self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    def __init__(self, device='cuda:0', max_length=1024,
                 checkpoint='vinai/bartpho-syllable-base',  # ← base로
                 use_word=False, cache_dir='/home/dxlab/backup/hf_cache'):  # ← 큰 디스크

         # ✅ 누락된 필드들 세팅
        self.device = device
        self.max_length = max_length
        self.use_word = use_word

        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, cache_dir=self.cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=self.cache_dir)

        self.model.eval().to(device)

        # pad_token_id 안전장치
        if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = 1  # BART 계열 기본값 보호

        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def _maybe_preprocess(self, xs):
        return preprocess_vi_word(xs) if self.use_word else xs

    def score(self, srcs, tgts, batch_size=4):
        scores = []
        srcs = self._maybe_preprocess(srcs)
        tgts = self._maybe_preprocess(tgts)

        for i in range(0, len(srcs), batch_size):
            src_batch = srcs[i:i+batch_size]
            tgt_batch = tgts[i:i+batch_size]
            with torch.no_grad():
                encoded_src = self.tokenizer(
                    src_batch, return_tensors='pt', padding=True, truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                encoded_tgt = self.tokenizer(
                    tgt_batch, return_tensors='pt', padding=True, truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                output = self.model(
                    input_ids=encoded_src['input_ids'],
                    attention_mask=encoded_src['attention_mask'],
                    labels=encoded_tgt['input_ids']
                )

                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), encoded_tgt['input_ids'].view(-1))
                loss = loss.view(encoded_tgt['input_ids'].shape[0], -1)
                # 길이 정규화
                denom = encoded_tgt['attention_mask'].sum(dim=1).clamp_min(1)
                normalized_loss = loss.sum(dim=1) / denom
                scores += [-l.item() for l in normalized_loss]
        return scores




def main():
    # 설정
    input_file = "/home/dxlab/data/dxlab/jupyter/heejeong/llm-generated_text_detection_v2/data/vietnamese_data/llm_regen2_data/regen2_1000.csv"  # CSV 파일 경로
    output_file = "/home/dxlab/data/dxlab/jupyter/heejeong/llm-generated_text_detection_v2/data/vietnamese_data/result_score/vietnamese_gpt35_simllm_score.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # 데이터 로드
    df = pd.read_csv(input_file)

    # ===== BARTpho 설정 =====
    # - 전처리 없이 시작: vinai/bartpho-syllable
    # - VnCoreNLP 분절/성조정규화 가능: vinai/bartpho-word + use_word=True
    checkpoint = "vinai/bartpho-syllable-base"
    scorer = ViBARTScorer(device=device, checkpoint=checkpoint, use_word=False)

    # 유사도 점수 계산
    print("🔍 Calculating LLM → LLM (regen1 → regen2) similarity...")
    sim_llm_score = scorer.score(df["regen1"].tolist(), df["regen2"].tolist())

    print("🔍 Calculating Human → LLM (human → regen1) similarity...")
    sim_human_score = scorer.score(df["human"].tolist(), df["regen1"].tolist())

    # 결과 저장
    df["sim_llm_score"] = sim_llm_score
    df["sim_human_score"] = sim_human_score
    df.to_csv(output_file, index=False)
    print(f"✅ Saved with both similarity scores to: {output_file}")


    from sklearn.metrics import roc_auc_score

    # ROC AUC 점수 계산
    scores = sim_llm_score + sim_human_score
    labels = [1] * len(sim_llm_score) + [0] * len(sim_human_score)
    roc_auc = roc_auc_score(labels, scores)
    print(f"📈 ROC AUC score: {roc_auc:.4f}")



if __name__ == "__main__":
    main()

