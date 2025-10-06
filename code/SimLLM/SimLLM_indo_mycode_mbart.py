import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from mbart_score_indo import MBARTScorer
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


class MBARTScorer:
    def __init__(
        self,
        device='cuda:0',
        max_length=1024,
        checkpoint='facebook/mbart-large-50-many-to-many-mmt',
        src_lang='id_ID',
        tgt_lang='id_ID'
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint).to(device)
        self.model.eval()

        bos_id = self.tokenizer.lang_code_to_id[tgt_lang]
        self.model.config.forced_bos_token_id = bos_id
        self.model.config.decoder_start_token_id = bos_id

        self.pad_id = self.model.config.pad_token_id
        self.lsm = nn.LogSoftmax(dim=-1)
        self.nll = nn.NLLLoss(reduction='none', ignore_index=self.pad_id)

    @torch.no_grad()
    def _neg_log_likelihood_per_token(self, src_list, tgt_list):

        encoded_src = self.tokenizer(
            src_list, return_tensors='pt', max_length=self.max_length,
            truncation=True, padding=True
        ).to(self.device)

        encoded_tgt = self.tokenizer(
            tgt_list, return_tensors='pt', max_length=self.max_length,
            truncation=True, padding=True
        ).to(self.device)

        outputs = self.model(
            input_ids=encoded_src['input_ids'],
            attention_mask=encoded_src['attention_mask'],
            labels=encoded_tgt['input_ids'],                      # ★ 중요
            decoder_attention_mask=encoded_tgt['attention_mask']  # 권장
        )
        logits = outputs.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :].contiguous()                       
        shift_labels = encoded_tgt['input_ids'][:, 1:].contiguous()           # [B, T-1]
        shift_mask   = encoded_tgt['attention_mask'][:, 1:].contiguous()      # [B, T-1]

        T = min(shift_logits.size(1), shift_labels.size(1), shift_mask.size(1))
        shift_logits = shift_logits[:, :T, :]
        shift_labels = shift_labels[:, :T]
        shift_mask   = shift_mask[:, :T]

        log_probs = self.lsm(shift_logits)                          # [B, T, V]
        gold_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                               # [B, T]

        token_counts = shift_mask.sum(dim=1).clamp(min=1)           # [B]
        nll = - (gold_log_probs * shift_mask).sum(dim=1) / token_counts
        return nll 


    def score(self, srcs, tgts, batch_size=4):
        scores = []
        for i in range(0, len(srcs), batch_size):
            src_batch = srcs[i:i+batch_size]
            tgt_batch = tgts[i:i+batch_size]
            nll = self._neg_log_likelihood_per_token(src_batch, tgt_batch) 
            scores += (-nll).tolist() 
        return scores

    def score_symmetric(self, a_list, b_list, batch_size=4):

        ab = self.score(a_list, b_list, batch_size=batch_size)
        ba = self.score(b_list, a_list, batch_size=batch_size)
        return [0.5*(x+y) for x, y in zip(ab, ba)]


device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device).eval()

tokenizer.src_lang = "id_ID"
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["id_ID"]
model.config.decoder_start_token_id = tokenizer.lang_code_to_id["id_ID"]

def main():
    input_file = "../../data/indo_news/llm_regen2_data/indo_regen2_1000_.csv" 
    output_file = "../../data/indo_news/result_score/indo_mbart_simllm_score.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(input_file)
    scorer = MBARTScorer(device=device)

    print("Calculating LLM → LLM similarity...")
    sim_gpt_gpt = scorer.score(df["regen1"].tolist(), df["regen2"].tolist())

    print("Calculating Human → LLM similarity...")
    sim_human_gpt = scorer.score(df["human"].tolist(), df["regen1"].tolist())

    df["sim_llm_scoret"] = sim_gpt_gpt
    df["sim_human_score"] = sim_human_gpt
    df.to_csv(output_file, index=False)
    print(f"Saved with both similarity scores to: {output_file}")

    from sklearn.metrics import roc_auc_score

    scores = sim_gpt_gpt + sim_human_gpt
    labels = [1] * len(sim_gpt_gpt) + [0] * len(sim_human_gpt)
    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC AUC score: {roc_auc:.4f}")



if __name__ == "__main__":
    main()

