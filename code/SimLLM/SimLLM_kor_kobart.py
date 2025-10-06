import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

class KoBARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='digit82/kobart-summarization'):
        self.device = device
        self.max_length = max_length
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval().to(device)
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        scores = []
        for i in range(0, len(srcs), batch_size):
            src_batch = srcs[i:i+batch_size]
            tgt_batch = tgts[i:i+batch_size]
            with torch.no_grad():
                encoded_src = self.tokenizer(src_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_length).to(self.device)
                encoded_tgt = self.tokenizer(tgt_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.max_length).to(self.device)

                output = self.model(input_ids=encoded_src['input_ids'],
                                    attention_mask=encoded_src['attention_mask'],
                                    labels=encoded_tgt['input_ids'])

                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), encoded_tgt['input_ids'].view(-1))
                loss = loss.view(encoded_tgt['input_ids'].shape[0], -1)
                normalized_loss = loss.sum(dim=1) / encoded_tgt['attention_mask'].sum(dim=1)
                scores += [-l.item() for l in normalized_loss]
        return scores


def main():
    input_file  = "../../data/kor_news/llm_regen2_data/gpt35_regen2_1000.csv"
    output_file = "../../data/kor_news/result_score/gpt35_distilkobert_simllm_score.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(input_file)
    scorer = KoBARTScorer(device=device)

    print("Calculating LLM → LLM (regen1 → regen2) similarity...")
    sim_llm_score = scorer.score(df["regen1"].tolist(), df["regen2"].tolist())

    print("Calculating Human → LLM (human → regen1) similarity...")
    sim_human_score = scorer.score(df["human"].tolist(), df["regen1"].tolist())

    df["sim_llm_score"] = sim_llm_score
    df["sim_human_score"] = sim_human_score
    df.to_csv(output_file, index=False)
    print(f"Saved with both similarity scores to: {output_file}")

    from sklearn.metrics import roc_auc_score

    scores = sim_llm_score + sim_human_score
    labels = [1] * len(sim_llm_score) + [0] * len(sim_human_score)
    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC AUC score: {roc_auc:.4f}")



if __name__ == "__main__":
    main()

