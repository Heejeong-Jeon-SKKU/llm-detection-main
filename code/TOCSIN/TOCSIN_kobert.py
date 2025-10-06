#!/usr/bin/env python
# TOCSIN_kobert.py
import os
import math
import time
import json
import tqdm
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_builder import generate_data, save_data, load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from kobart_score import KoBARTScorer as BARTScorer  # perturbation 유사도는 그대로 KoBART 사용

# ----------------------------------------
# 전역 설정
# ----------------------------------------
pct = 0.015
# TOK_ARGS = dict(return_tensors="pt", padding=True, return_token_type_ids=False, max_length=512, truncation=True)
TOK_ARGS = dict(return_tensors="pt", padding=True, return_token_type_ids=False, max_length=384, truncation=True)


# ----------------------------------------
# Perturbation (원 코드 유지)
# ----------------------------------------
def fill_and_mask(text, pct=pct):
    tokens = text.split(' ')
    n_spans = int(pct * len(tokens))
    if n_spans <= 0:
        return []
    idxs = np.random.choice(range(len(tokens)), size=n_spans, replace=False)
    return idxs.tolist()

def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]
    for t_idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for i in indices:
            if 0 <= i < len(text):
                text[i] = ""
    return [" ".join(x) for x in tokens]

def perturb_texts_(texts, pct=pct):
    indices_list = [fill_and_mask(x, pct) for x in texts]
    return apply_extracted_fills(texts, indices_list)

def perturb_texts(texts, pct=pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs

# ----------------------------------------
# KoBERT(MLM) → PLL logits(1, T, V) 생성
# ----------------------------------------
@torch.no_grad()
def kobert_pll_logits(model, tokenizer, text, device, chunk_size=16):
    SAFE = "."
    if not isinstance(text, str) or text.strip() == "":
        text = SAFE

    enc = tokenizer(text, **TOK_ARGS)
    input_ids = enc["input_ids"].to(device)      # (1, T)
    attn      = enc["attention_mask"].to(device) # (1, T)
    if input_ids.size(1) == 0:
        enc = tokenizer(SAFE, **TOK_ARGS)
        input_ids = enc["input_ids"].to(device)
        attn      = enc["attention_mask"].to(device)

    T = input_ids.size(1)
    V = model.config.vocab_size
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "KoBERT tokenizer must have a [MASK] token."

    # 결과 담을 버퍼 (1, T, V)
    out_logits = torch.empty((1, T, V), device=device, dtype=torch.float32)

    positions = torch.arange(T, device=device)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        pos = positions[start:end]                # (B,)
        B = pos.size(0)

        # (B, T) 복제 후 각 행의 해당 위치를 [MASK]
        input_rep = input_ids.repeat(B, 1)        # (B, T)
        attn_rep  = attn.repeat(B, 1)             # (B, T)
        input_rep[torch.arange(B, device=device), pos] = mask_id

        # (선택) 반정밀도로 메모리/속도 절감
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            pred = model(input_ids=input_rep, attention_mask=attn_rep).logits  # (B, T, V)

        # 각 행에서 '그 마스킹한 위치'의 분포만 추출해 저장
        # out_logits[0, pos, :] = pred[torch.arange(B, device=device), pos, :]
        out_logits[0, pos, :] = pred[torch.arange(B, device=device), pos, :].to(out_logits.dtype)


        # 메모리 조절
        del pred, input_rep, attn_rep
        torch.cuda.empty_cache()

    return out_logits, input_ids

# ----------------------------------------
# 점수 계산 유틸 (원 코드와 호환)
# ----------------------------------------
def get_samples(logits, nsamples=4000):
    # logits: (1, T, V)
    assert logits.dim() == 3 and logits.size(0) == 1
    lprobs = torch.log_softmax(logits, dim=-1)            # (1, T, V)
    distrib = torch.distributions.Categorical(logits=lprobs)  # (1, T, V)
    # sample shape: (nsamples, 1, T)
    samples = distrib.sample([nsamples]).permute(1, 2, 0)      # (1, T, nsamples)
    return samples

def get_likelihood(logits, labels):
    # logits: (1, T, V), labels: (1, T) or (1, T, K)
    assert logits.size(0) == 1
    lprobs = torch.log_softmax(logits, dim=-1)
    if labels.dim() == 2:
        labels = labels.unsqueeze(-1)                  # (1, T, 1)
    ll = lprobs.gather(dim=-1, index=labels)          # (1, T, K)
    return ll.mean(dim=1)                              # (1, K)

def get_logrank(logits, labels):
    # logits: (1, T, V), labels: (1, T)
    assert logits.size(0) == 1 and labels.size(0) == 1
    order = logits.argsort(dim=-1, descending=True)   # (1, T, V)
    matches = (order == labels.unsqueeze(-1)).nonzero()
    # 느슨하게만 체크(각 timestep에 하나씩 있어야 함)
    assert matches.size(1) == 3, f"Unexpected matches shape: {matches.shape}"
    ranks = matches[:, -1].float() + 1.0
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_score(logits_ref, logits_score, labels, source_texts, perturbed_texts, basemodel, bart_scorer):
    # vocab 정렬
    if logits_ref.size(-1) != logits_score.size(-1):
        V = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref   = logits_ref[:, :, :V]
        logits_score = logits_score[:, :, :V]

    # 길이 정렬
    T = min(logits_ref.size(1), logits_score.size(1), labels.size(1))
    logits_ref   = logits_ref[:, :T]
    logits_score = logits_score[:, :T]
    labels       = labels[:, :T]

    # 샘플 + 인덱스 가드
    samples_2 = get_samples(logits_ref, nsamples=4000)
    V = logits_score.size(-1)
    samples_2 = samples_2.clamp(0, V - 1)

    log_likelihood_x       = get_likelihood(logits_score, labels)
    log_rank_x             = get_logrank(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_2)
    miu_tilde   = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    # KoBARTScorer로 perturbation 유사도(기존과 동일)
    source_texts_list = [source_texts] * 10
    values = bart_scorer.score(perturbed_texts, source_texts_list, batch_size=10)
    mean_values = np.mean(values)

    if basemodel == 'Fast':
        denom = max(sigma_tilde.item(), 1e-8)
        output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / denom) * math.exp(-mean_values)
    elif basemodel == 'lrr':
        output_score = (log_likelihood_x.squeeze(-1).item() / max(log_rank_x, 1e-8)) * math.exp(-mean_values)
    elif basemodel == 'likelihood':
        output_score =  log_likelihood_x.squeeze(-1).item() * math.exp(mean_values)
    elif basemodel == 'logrank':
        output_score =  log_rank_x * math.exp(mean_values)
    elif basemodel == 'standalone':
        output_score = -mean_values
    else:
        output_score =  log_likelihood_x.squeeze(-1).item() * math.exp(mean_values)

    return output_score

# ----------------------------------------
# 메인 실험 루프
# ----------------------------------------
def experiment(args):
    device = args.device

    # KoBERT 로딩 (scoring/ref)
    scoring_tokenizer = AutoTokenizer.from_pretrained(args.scoring_model_name, cache_dir=args.cache_dir, use_fast=True)
    scoring_model     = AutoModelForMaskedLM.from_pretrained(args.scoring_model_name, cache_dir=args.cache_dir).to(device).eval()

    if args.reference_model_name == args.scoring_model_name:
        reference_tokenizer = scoring_tokenizer
        reference_model     = scoring_model
    else:
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model_name, cache_dir=args.cache_dir, use_fast=True)
        reference_model     = AutoModelForMaskedLM.from_pretrained(args.reference_model_name, cache_dir=args.cache_dir).to(device).eval()

    # 데이터
    if args.skip_generation:
        data = load_data(args.dataset_file)
    else:
        dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document', 'kor_dataset': 'original'}
        key = dataset_keys[args.dataset] if args.dataset in dataset_keys else None
        data = generate_data(args, args.dataset, key)
        save_data(args.dataset_file, args, data)

    n_originals = len(data['original'])
    n_samples   = len(data['sampled'])
    name        = args.basemodel
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # KoBARTScorer (유사도)
    bart_scorer = BARTScorer(device=device, checkpoint='digit82/kobart-summarization')

    # Perturb를 미리 만들어 batched scoring에 사용
    perturbed_original_texts = perturb_texts([x for x in data['original'] for _ in range(10)])
    perturbed_sampled_texts  = perturb_texts([x for x in data['sampled']  for _ in range(10)])
    perturbed_original_texts_list = []
    perturbed_sampled_texts_list  = []
    for idx in range(n_originals):
        perturbed_original_texts_list.append(perturbed_original_texts[idx * 10: (idx + 1) * 10])
        perturbed_sampled_texts_list.append( perturbed_sampled_texts[idx * 10: (idx + 1) * 10])

    # 루프
    results = []
    start_time = time.time()
    for idx in tqdm.tqdm(range(n_originals), desc="computing"):
        original_text = data["original"][idx] or "."
        sampled_text  = data["sampled"][idx]  or "."

        # KoBERT PLL logits/labels
        logits_score, labels = kobert_pll_logits(scoring_model, scoring_tokenizer, original_text, device)
        if args.reference_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            logits_ref, _ = kobert_pll_logits(reference_model, reference_tokenizer, original_text, device)

        original_crit = get_score(
            logits_ref, logits_score, labels, original_text,
            perturbed_original_texts_list[idx], args.basemodel, bart_scorer
        )

        logits_score_s, labels_s = kobert_pll_logits(scoring_model, scoring_tokenizer, sampled_text, device)
        if args.reference_model_name == args.scoring_model_name:
            logits_ref_s = logits_score_s
        else:
            logits_ref_s, _ = kobert_pll_logits(reference_model, reference_tokenizer, sampled_text, device)

        sampled_crit = get_score(
            logits_ref_s, logits_score_s, labels_s, sampled_text,
            perturbed_sampled_texts_list[idx], args.basemodel, bart_scorer
        )

        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })

    # 메트릭/저장
    predictions = {
        'real':    [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"]  for x in results]
    }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc      = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # 파일명/저장 경로
    def sanitize_model_name(nm): return nm.split('/')[-1].replace('-', '_')
    ref_name    = sanitize_model_name(args.reference_model_name)
    score_name  = sanitize_model_name(args.scoring_model_name)
    dataset_nm  = args.dataset.replace('-', '_')
    output_filename = f'{dataset_nm}_{score_name}_{args.basemodel}.json'
    os.makedirs(args.output_file_dir, exist_ok=True)
    results_file = os.path.join(args.output_file_dir, output_filename)

    out = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'loss': 1 - pr_auc
    }
    with open(results_file, 'w') as fout:
        json.dump(out, fout)
        print(f'Results written into {results_file}')

    print(f"cost of time: {time.time() - start_time:.2f}s")
    return roc_auc

# ----------------------------------------
# Entrypoint
# ----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_dir', type=str, default="./results/")
    parser.add_argument('--dataset', type=str, default="kor_dataset")
    parser.add_argument('--dataset_file', type=str, default="data_test/")
    parser.add_argument('--reference_model_name', type=str, default="skt/kobert-base-v1")
    parser.add_argument('--scoring_model_name', type=str, default="skt/kobert-base-v1")
    parser.add_argument('--basemodel', type=str, default="Fast")  # Fast / lrr / likelihood / logrank / standalone
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--skip_generation', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    experiment(args)
