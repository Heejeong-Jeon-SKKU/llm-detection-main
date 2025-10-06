import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from kobart_score import KoBARTScorer
from distilkobert_score import DistilKoBERTScorer


class DistilKoBERTScorer:
    def __init__(
        self,
        device: str = "cuda:0",
        checkpoint: str = "monologg/distilkobert",
        max_length: int = 256,
        chunk_size: int = 64,
        amp_dtype = torch.float16,
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = False, 
    ):
        self.device = device
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.amp_dtype = amp_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()

        assert self.tokenizer.mask_token_id is not None, "Tokenizer must have a [MASK] token."
        assert self.tokenizer.sep_token_id is not None and self.tokenizer.cls_token_id is not None, \
            "Tokenizer must have [CLS]/[SEP] tokens."

        self.lsm = nn.LogSoftmax(dim=-1)

    @torch.no_grad()
    def _score_pair(self, src: str, tgt: str) -> float:
        if not isinstance(src, str) or src.strip() == "":
            src = "."
        if not isinstance(tgt, str) or tgt.strip() == "":
            tgt = "."

        enc = self.tokenizer(
            src, tgt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = enc["input_ids"].to(self.device)         
        attn      = enc["attention_mask"].to(self.device)    
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        T = input_ids.size(1)
        ids = input_ids[0]                                   
        sep_id = self.tokenizer.sep_token_id

        sep_positions = (ids == sep_id).nonzero(as_tuple=False).view(-1).tolist()
        if len(sep_positions) >= 2:
            src_end = sep_positions[0]
            tgt_end = sep_positions[1]
            tgt_start = src_end + 1
            tgt_positions = list(range(tgt_start, tgt_end))
        else:
            tgt_positions = list(range(T // 2, max(T - 1, T // 2)))

        if len(tgt_positions) == 0:
            return 0.0

        total_nll = 0.0
        total_cnt = 0
        use_amp = self.device.startswith("cuda")

        for s in range(0, len(tgt_positions), self.chunk_size):
            pos_chunk = tgt_positions[s:s + self.chunk_size]
            B = len(pos_chunk)

            input_rep = input_ids.repeat(B, 1)
            attn_rep  = attn.repeat(B, 1)

            row_idx = torch.arange(B, device=self.device)
            col_idx = torch.tensor(pos_chunk, device=self.device)
            input_rep[row_idx, col_idx] = self.tokenizer.mask_token_id

            if use_amp:
                with torch.amp.autocast('cuda', enabled=True, dtype=self.amp_dtype):
                    logits = self.model(input_ids=input_rep, attention_mask=attn_rep).logits 
            else:
                logits = self.model(input_ids=input_rep, attention_mask=attn_rep).logits

            lprobs = self.lsm(logits)  
            target_tokens = ids[col_idx] 
            selected = lprobs[row_idx, col_idx, target_tokens]
            total_nll += (-selected).sum().item()
            total_cnt += B

            del logits, lprobs, input_rep, attn_rep

        avg_nll = total_nll / max(total_cnt, 1)
        return -avg_nll 

    def score(self, srcs, tgts, batch_size: int = 1):
        scores = []
        for s, t in zip(srcs, tgts):
            scores.append(self._score_pair(s, t))
        return scores


def main():
    input_file  = "../../data/kor_news/llm_regen2_data/gpt35_distilkobert_regen2_1000.csv"
    output_file = "../../data/kor_news/result_score/gpt35_distilkobert_simllm_score.csv"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(input_file)

    scorer = DistilKoBERTScorer(
        device=device,
        checkpoint="monologg/distilkobert",
        max_length=256,          
        chunk_size=64,           
        trust_remote_code=True,
        use_fast_tokenizer=False 
    )

    print("Calculating LLM → LLM (regen1 → regen2) similarity...")
    sim_llm_score = scorer.score(df["regen1"].tolist(), df["regen2"].tolist())

    print("Calculating Human → LLM (human → regen1) similarity...")
    sim_human_score = scorer.score(df["human"].tolist(), df["regen1"].tolist())

    df["sim_llm_score"]   = sim_llm_score
    df["sim_human_score"] = sim_human_score
    df.to_csv(output_file, index=False)
    print(f"Saved with both similarity scores to: {output_file}")

    scores = sim_llm_score + sim_human_score
    labels = [1] * len(sim_llm_score) + [0] * len(sim_human_score)
    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC AUC score: {roc_auc:.4f}")

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
