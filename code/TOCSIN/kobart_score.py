# kobart_score.py
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

class KoBARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='digit82/kobart-summarization'):
        self.device = device
        self.max_length = max_length
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    # def score(self, srcs, tgts, batch_size=4):
    #     score_list = []
    #     for i in range(0, len(srcs), batch_size):
    #         src_list = srcs[i:i + batch_size]
    #         tgt_list = tgts[i:i + batch_size]
    #         with torch.no_grad():
    #             encoded_src = self.tokenizer(src_list, return_tensors='pt', max_length=self.max_length,
    #                                          truncation=True, padding=True)
    #             encoded_tgt = self.tokenizer(tgt_list, return_tensors='pt', max_length=self.max_length,
    #                                          truncation=True, padding=True)

    #             src_tokens = encoded_src['input_ids'].to(self.device)
    #             src_mask = encoded_src['attention_mask'].to(self.device)
    #             tgt_tokens = encoded_tgt['input_ids'].to(self.device)
    #             tgt_mask = encoded_tgt['attention_mask']
    #             tgt_len = tgt_mask.sum(dim=1).to(self.device)

    #             output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
    #             logits = output.logits.view(-1, self.model.config.vocab_size)
    #             loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
    #             loss = loss.view(tgt_tokens.shape[0], -1)
    #             loss = loss.sum(dim=1) / tgt_len
    #             curr_score_list = [-x.item() for x in loss]
    #             score_list += curr_score_list
    #     return score_list
    def score(self, srcs, tgts, batch_size=4):
        score_list = []
        SAFE = "."  # (1) 빈 문자열 대체용 안전 토큰

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i + batch_size]
            tgt_list = tgts[i:i + batch_size]

            # (1) 입력 단계에서 빈/공백 치환
            src_list = [s if isinstance(s, str) and s.strip() != "" else SAFE for s in src_list]
            tgt_list = [t if isinstance(t, str) and t.strip() != "" else SAFE for t in tgt_list]

            with torch.no_grad():
                encoded_src = self.tokenizer(
                    src_list, return_tensors='pt', max_length=self.max_length,
                    truncation=True, padding=True
                )
                encoded_tgt = self.tokenizer(
                    tgt_list, return_tensors='pt', max_length=self.max_length,
                    truncation=True, padding=True
                )

                # (2) 토크나이즈 후에도 혹시 길이 0이면 안전 문자열로 재토크나이즈
                if encoded_src['input_ids'].size(1) == 0:
                    encoded_src = self.tokenizer(
                        [SAFE] * len(src_list), return_tensors='pt', max_length=self.max_length,
                        truncation=True, padding=True
                    )
                if encoded_tgt['input_ids'].size(1) == 0:
                    encoded_tgt = self.tokenizer(
                        [SAFE] * len(tgt_list), return_tensors='pt', max_length=self.max_length,
                        truncation=True, padding=True
                    )

                src_tokens = encoded_src['input_ids'].to(self.device)
                src_mask = encoded_src['attention_mask'].to(self.device)
                tgt_tokens = encoded_tgt['input_ids'].to(self.device)

                # (3) 분모 0 방지: attention_mask가 있으면 쓰고, 없거나 전부 0이면 pad 기준으로 대체
                tgt_mask = encoded_tgt.get('attention_mask', (encoded_tgt['input_ids'] != self.model.config.pad_token_id).long())
                tgt_len = tgt_mask.sum(dim=1).to(self.device).clamp_min(1)

                output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)

                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss.sum(dim=1) / tgt_len

                score_list += [-x.item() for x in loss]

        return score_list
