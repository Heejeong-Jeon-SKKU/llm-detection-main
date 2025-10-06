echo `date`, Setup the environment ...
set -e

# 디렉토리 생성
exp_path=exp_Korean_TOCSIN
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

export CUDA_VISIBLE_DEVICES=1

# 실험 설정
ref_model='digit82/kobart-summarization' 
score_model='digit82/kobart-summarization'
dataset='kor_dataset'
base_model='standalone'
dataset_file_path='../../data/kor_news/llm_cont_data/gpt35_cont_1000_merged.json'

echo `date`, Evaluating on ${dataset}.${ref_model}_${score_model} ...
python ./TOCSIN_kobart.py \
  --reference_model_name $ref_model \
  --scoring_model_name $score_model \
  --basemodel $base_model \
  --dataset $dataset \
  --dataset_file $dataset_file_path \
  --skip_generation \
  --output_file $res_path/tocsin_gpt35_kobart_1000_result \
  --cache_dir ./pretrained_models \
