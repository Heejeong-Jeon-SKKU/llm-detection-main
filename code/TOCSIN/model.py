# Copyright (c) Guangsheng Bao.
# This source code is licensed under the MIT license.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# ✅ 1. 사용할 모델들 이름 정의
model_fullnames = {
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'koalpaca': 'beomi/KoAlpaca-Polyglot-7B',
}

# ✅ 2. float16이 필요한 모델 명시
float16_models = ['llama3-8b', 'koalpaca']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name


def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')

    # 양자화 설정 (4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16  # or torch.float16 depending on GPU
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_fullname,
        quantization_config=bnb_config,
        device_map="auto",  # GPU 자동 분배
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    return model

# def load_model(model_name, device, cache_dir):
#     model_fullname = get_model_fullname(model_name)
#     print(f'Loading model {model_fullname}...')
#     model_kwargs = {}

#     # ✅ 3. float16 설정
#     if model_name in float16_models:
#         model_kwargs.update(dict(torch_dtype=torch.float16))

#     model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
#     print('Moving model to GPU...', end='', flush=True)
#     start = time.time()
#     model.to(device)
#     print(f'DONE ({time.time() - start:.2f}s)')
#     return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}

    # ✅ 4. padding 설정
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'

    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)

    # ✅ 5. pad_token_id 설정
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_tokenizer

# ✅ 테스트 실행용
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-8b")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
