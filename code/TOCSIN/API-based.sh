#!/usr/bin/env bash

echo `date`, Setup the environment ...
set -e

# 디렉토리 생성
exp_path=exp_Korean_TOCSIN
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# GPU 메모리 파편화 방지 설정
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (선택) 특정 GPU만 사용하고 싶을 경우 아래 주석 해제
export CUDA_VISIBLE_DEVICES=1

# 실험 설정
ref_model='facebook/bart-base' # 이미 데이터셋에 llm생성 문장 준비되어있음 
score_model='facebook/bart-base'
dataset='kor_dataset'
base_model='standalone' # bartscore만 단독 사용(loglikelihood/logrank 계산 안함)
dataset_file_path='/home/dxlab/data/dxlab/jupyter/heejeong/llm-generated_text_detection_v2/data/klue/llm_cont_data/klue_cont_gpt35_1000_merged.json'

echo `date`, Evaluating on ${dataset}.${ref_model}_${score_model} ...
# python ./TOCSIN_kobart_revised.py \
python ./TOCSIN_kobart.py \
  --reference_model_name $ref_model \
  --scoring_model_name $score_model \
  --basemodel $base_model \
  --dataset $dataset \
  --dataset_file $dataset_file_path \
  --skip_generation \
  --output_file $res_path/tocsin_klue_gpt35_kobart_1000_result \
  --cache_dir /home/dxlab/backup

# --output_file $res_path/${dataset}.${ref_model}_${score_model} \