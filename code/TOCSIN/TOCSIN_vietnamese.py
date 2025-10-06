
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import datasets
import torch.nn.functional as F
import tqdm
import argparse
import json
from model import load_tokenizer, load_model
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import scipy
import math
import jsonlines
# from bart_score import BARTScorer
# from kobart_score import KoBARTScorer as BARTScorer 
from bart_score import BARTScorer
import matplotlib.pyplot as plt
import time
from data_builder import generate_data, save_data, load_data
import unicodedata, regex as re


pct = 0.015

def save_llr_histograms(prediction, name):
    plt.clf()
    plt.figure(figsize=(8,5))
    results = {'human-written':[],'LLM-generated':[]}
    for r in range(len(prediction['real'])):
        results['human-written'].append(prediction["real"][r])
        results['LLM-generated'].append(prediction["samples"][r])
    if name == "gpt2-xl":
        plt.hist([r for r in results['human-written']], alpha=0.5, bins=130, range=(0, 1.60), label='Human-written')
        plt.hist([r for r in results['LLM-generated']], alpha=0.5, bins=75, range=(0, 1.60), label='LLM-generated')
        plt.legend(loc='upper right',fontsize=21)
        plt.ylim(0, 100)
        plt.xticks(fontsize=20)
        plt.yticks([0, 20, 40, 60, 80, 100],fontsize=20)
    else:
        plt.hist([r for r in results['human-written']], alpha=0.5, bins=130, range=(0, 1.60))
        plt.hist([r for r in results['LLM-generated']], alpha=0.5, bins=75, range=(0, 1.60))
        plt.ylim(0, 100)
        plt.xticks(fontsize=20)
        plt.yticks([0, 20, 40, 60, 80, 100],fontsize=20)


    plt.savefig(f"sim_histograms_{name}.png")


def fill_and_mask(text,  pct = pct):
    tokens = text.split(' ')

    n_spans = pct * len(tokens)
    n_spans = int(n_spans)

    repeated_random_numbers = np.random.choice(range(len(tokens)), size=n_spans)

    return repeated_random_numbers.tolist()


def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]

    for idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for idx in indices:
            text[idx] = ""

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, pct=pct):

    indices_list = [fill_and_mask(x, pct) for x in texts]
    perturbed_texts = apply_extracted_fills(texts, indices_list)

    return perturbed_texts


def perturb_texts(texts, pct=pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs

# def perturb_texts_tokenaware(texts, tokenizer, pct=0.08, batch=64):
#     out = []
#     for i in range(0, len(texts), batch):
#         batch_txt = texts[i:i+batch]
#         enc = tokenizer(batch_txt, padding=False, truncation=True, max_length=512)
#         for ids in enc["input_ids"]:
#             n = max(1, int(len(ids) * pct))
#             idx = np.random.choice(range(1, len(ids)-1), size=n, replace=False)  # BOS/EOS 제외
#             kept = [t for j,t in enumerate(ids) if j not in idx]
#             out.append(tokenizer.decode(kept, skip_special_tokens=True))
#     return out
def perturb_texts_tokenaware(texts, tokenizer, pct=0.08, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        batch_txt = texts[i:i+batch]
        enc = tokenizer(batch_txt, padding=False, truncation=True, max_length=512)
        for ids in enc["input_ids"]:
            if len(ids) <= 2:
                out.append(tokenizer.decode(ids, skip_special_tokens=True))
                continue
            n_total = len(ids) - 2  # BOS/EOS 제외 가능 토큰 수
            n = max(1, int(len(ids) * pct))
            n = min(n, n_total)
            if n <= 0:
                out.append(tokenizer.decode(ids, skip_special_tokens=True))
                continue
            idx = np.random.choice(range(1, len(ids)-1), size=n, replace=False)
            kept = [t for j,t in enumerate(ids) if j not in idx]
            out.append(tokenizer.decode(kept, skip_special_tokens=True))
    return out



def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples_2 = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples_2

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def bart_mean_symmetric(srcs, tgts, bs=10):
    # 양방향 BARTScore 계산 후, 각 쌍의 평균을 구하고 다시 전체 평균
    s1 = bart_scorer.score(srcs, tgts, batch_size=bs)  # src→tgt
    s2 = bart_scorer.score(tgts, srcs, batch_size=bs)  # tgt→src
    return float(np.mean([(a + b) / 2 for a, b in zip(s1, s2)]))


def get_score(logits_ref, logits_score, labels,source_texts, perturbed_texts, basemodel):
    alpha = getattr(args, 'alpha', 1.0)

    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    
    # --- vocab 정렬 ---
    if logits_ref.size(-1) != logits_score.size(-1):
        V = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref   = logits_ref[:, :, :V]
        logits_score = logits_score[:, :, :V]

    # 🔧 --- time 정렬 (반드시 추가) ---
    T = min(logits_ref.size(1), logits_score.size(1), labels.size(1))
    logits_ref   = logits_ref[:, :T, :]
    logits_score = logits_score[:, :T, :]
    labels       = labels[:, :T]

    samples_2 = get_samples(logits_ref, labels)

    log_likelihood_x = get_likelihood(logits_score, labels)
    log_rank_x = get_logrank(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_2)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    source_texts_list = [source_texts]*10

    #bart-score
    values = bart_scorer.score(perturbed_texts,source_texts_list, batch_size=10)
    # mean_values = np.mean(values)
    mean_values = bart_mean_symmetric(perturbed_texts, source_texts_list, bs=10)
    
    # if basemodel == 'Fast':
    #     if 'gemini' in args.dataset_file and 'pubmed' in args.dataset_file:
    #         #Fast-DetectGPT has too many negative values in the output scores on the data generated by Gemini on PubMed, which can lead to scaling of the improvement effect. A constant is added here to address this.
    #         output_score = (((log_likelihood_x.squeeze(-1).item()  - miu_tilde.item())/ (sigma_tilde.item()))+2) * math.pow(math.e, -mean_values)
    #     else:
    #         output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / (sigma_tilde.item())) * math.pow(math.e, -mean_values)
    # elif basemodel == 'lrr':
    #     output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, -mean_values)
    # elif basemodel == 'likelihood':
    #     output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)
    # elif basemodel == 'logrank':
    #     output_score = log_rank_x * math.pow(math.e, mean_values)
    # elif basemodel == 'standalone':
    #     output_score = -mean_values
    if basemodel == 'Fast':
        z = (log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / max(1e-6, sigma_tilde.item())
        output_score = z * math.exp(-alpha * mean_values)   # ← 부호/α 스윕
    elif basemodel == 'likelihood':
        output_score = log_likelihood_x.squeeze(-1).item() * math.exp(alpha * mean_values)
    elif basemodel == 'standalone':
        output_score = -mean_values

    #The default base model is set to likelihood
    else:
        output_score =  log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)

    return output_score

def experiment():

    # load model
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset,args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()

    # data = load_data(args.dataset_file)
    # ✅ 아래 코드로 대체
    if args.skip_generation:
        data = load_data(args.dataset_file)
    else:
        dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document', 'kor_dataset': 'original'}
        key = dataset_keys[args.dataset] if args.dataset in dataset_keys else None
        data = generate_data(args, args.dataset, key)
        save_data(args.dataset_file, args, data)
    
    
    def normalize_vi(s):
        s = unicodedata.normalize('NFC', s)       # 성조 결합형 정규화
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # 데이터 로딩 직후
    data['original'] = [normalize_vi(x) for x in data['original']]
    data['sampled']  = [normalize_vi(x) for x in data['sampled']]


    n_originals = len(data['original'])
    n_samples = len(data["sampled"])
    start_time = time.time()
    # evaluate criterion

    name = args.basemodel
    criterion_fn = get_score

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []

    # perturbed_original_texts = perturb_texts([x for x in data['original'] for _ in range(10)])
    # perturbed_sampled_texts = perturb_texts([x for x in data['sampled'] for _ in range(10)])
    perturbed_original_texts = perturb_texts_tokenaware(
        [x for x in data['original'] for _ in range(10)], scoring_tokenizer, pct=0.08
    )
    perturbed_sampled_texts = perturb_texts_tokenaware(
        [x for x in data['sampled'] for _ in range(10)], scoring_tokenizer, pct=0.08
    )


    perturbed_original_texts_list = []
    perturbed_sampled_texts_list = []

    for idx in range(len(data['original'])):
        perturbed_original_texts_list.append(perturbed_original_texts[idx * 10: (idx + 1) * 10])
        perturbed_sampled_texts_list.append(perturbed_sampled_texts[idx * 10: (idx + 1) * 10])


    for idx in tqdm.tqdm(range(n_originals),desc="computing"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False, max_length=512, truncation=True).to(args.device)

        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():

            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                # tokenized  = scoring_tokenizer(original_text,  return_tensors="pt", padding=True,
                #                   return_token_type_ids=False, max_length=512, truncation=True).to(args.device)
                # assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                # logits_ref = reference_model(**tokenized).logits[:, :-1]
                tokenized_ref = reference_tokenizer(original_text, return_tensors="pt", padding=True,
                                    return_token_type_ids=False, max_length=512, truncation=True).to(args.device)
                logits_ref = reference_model(**tokenized_ref).logits[:, :-1]
            
            original_crit = criterion_fn(logits_ref, logits_score, labels, original_text,perturbed_original_texts_list[idx], args.basemodel)

        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                # tokenized     = scoring_tokenizer(sampled_text,  return_tensors="pt", padding=True,
                                #   return_token_type_ids=False, max_length=512, truncation=True).to(args.device)
                # assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                # logits_ref = reference_model(**tokenized).logits[:, :-1]
                tokenized_ref = reference_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                    return_token_type_ids=False, max_length=512, truncation=True).to(args.device)
                logits_ref = reference_model(**tokenized_ref).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, sampled_text, perturbed_sampled_texts_list[idx], args.basemodel)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results 저장 
    def sanitize_filename(s):
        return s.replace('/', '_')
    
    def sanitize_model_name(name):
        return name.split('/')[-1].replace('-', '_')

    # ref_name_safe = sanitize_filename(args.reference_model_name)
    # score_name_safe = sanitize_filename(args.scoring_model_name)
    # dataset_safe = sanitize_filename(args.dataset)
    # output_file_base = f'{dataset_safe}.{ref_name_safe}_{score_name_safe}'
    # results_file = f'{args.output_file_dir}/{output_file_base}.{name}.json'

    ref_name = sanitize_model_name(args.reference_model_name)
    score_name = sanitize_model_name(args.scoring_model_name)
    dataset_name = args.dataset.replace('-', '_')
    output_filename = f'{dataset_name}_{score_name}_{args.basemodel}.json'
    results_file = os.path.join(args.output_file_dir, output_filename)

    # results_file = f'{args.output_file}.{name}.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    results = {'name': f'{name}_threshold',
               'info': {'n_samples': n_samples},
               'predictions': predictions,
               'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
               'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

    #save_llr_histograms(predictions,args.scoring_model_name)
    end_time = time.time()
    print(f"cost of time：{args.scoring_model_name}",end_time - start_time)
    return roc_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_file', type=str, default="./results_/")
    parser.add_argument('--output_file_dir', type=str, default="./results/")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="data_test/")
    parser.add_argument('--reference_model_name', type=str, default="gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--basemodel', type=str, default="Fast")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--skip_generation', action='store_true') # 이미 생성된 데이터가 있어서, 생성 단계를 건너뜀
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    base_tokenizer = scoring_tokenizer

    DEVICE = "cuda:0"
    # bart_scorer = BARTScorer(device=DEVICE, checkpoint='facebook/bart-base')
    # bart_scorer = BARTScorer(device="cpu", checkpoint='facebook/bart-base')
    # ✅ 수정: KoBART 사용
    # bart_scorer = BARTScorer(device=args.device, checkpoint='digit82/kobart-summarization')

    # VIETNAMESE BART 사용 
    VI_CHECKPOINT = "vinai/bartpho-syllable-base"  # 또는 "vinai/bartpho-word"
    bart_scorer = BARTScorer(device=args.device, checkpoint=VI_CHECKPOINT)

    experiment()

