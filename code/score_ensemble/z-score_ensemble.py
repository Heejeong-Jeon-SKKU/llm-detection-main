# TOCSIN / SimLLM score ensemble (z-score based)

import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
    roc_auc_score
)
from scipy.special import softmax, expit
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 30  
plt.rcParams['axes.titlesize'] = 42
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28


SAVE_DIR = './result/ensemble_gpt35_1000.csv/'
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv('./scores/gpt35_scores_1000.csv') 

# tuning set/evaluation set split (8:2 stratified)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# tuning set 기준 z-score 정규화
def zscore_train_based(train_series, test_series):
    mean = train_series.mean()
    std = train_series.std()
    return (train_series - mean) / std, (test_series - mean) / std

df_train['tocsin_z'], df_test['tocsin_z'] = zscore_train_based(df_train['tocsin_score'], df_test['tocsin_score'])
df_train['simllm_z'], df_test['simllm_z'] = zscore_train_based(df_train['simllm_score'], df_test['simllm_score'])

# 앙상블 점수 계산
def compute_ensemble_features(df):
    z_mat = np.stack([df['tocsin_z'].values, df['simllm_z'].values], axis=1)
    df['ensemble_z_weighted'] = 0.8 * df['tocsin_z'] + 0.2 * df['simllm_z']
    df['ensemble_softmax'] = softmax(z_mat, axis=1)[:, 1]
    df['ensemble_z_mean'] = (df['tocsin_z'] + df['simllm_z']) / 2
    df['ensemble_sigmoid'] = expit(df['ensemble_z_mean'])
    return df

df_train = compute_ensemble_features(df_train)
df_test = compute_ensemble_features(df_test)

# threshold 저장용 딕셔너리 (각 점수 컬럼별로 sweep과 kde 임계값 저장)
threshold_dict = {} 

# draw figures
def draw_hist(df_sub, col_name, title, fname, label_text):
    lower = df_sub[col_name].quantile(0.005)
    upper = df_sub[col_name].quantile(0.995)
    df_filtered = df_sub[(df_sub[col_name] >= lower) & (df_sub[col_name] <= upper)]

    plt.figure(figsize=(16, 7))
    sns.histplot(df_filtered[df_filtered['label'] == 0][col_name], color='tab:blue', label='Human', kde=True, stat='density', bins=30, alpha=0.7)
    sns.histplot(df_filtered[df_filtered['label'] == 1][col_name], color='tab:red', label='LLM', kde=True, stat='density', bins=30, alpha=0.7)
    
    if col_name in threshold_dict:
        if 'sweep' in threshold_dict[col_name]:
            thr_sweep = threshold_dict[col_name]['sweep']
            plt.axvline(thr_sweep, color='forestgreen', linestyle='--', label=f'Sweep={thr_sweep:.3f}')
        if 'kde' in threshold_dict[col_name]:
            thr_kde = threshold_dict[col_name]['kde']
            plt.axvline(thr_kde, color='orange', linestyle='--', label=f'KDE={thr_kde:.3f}')
    
    plt.title(title)
    plt.xlabel(label_text)
    plt.ylabel("Density")
    plt.legend(loc='upper left', framealpha=0.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

def draw_hist_right(df_sub, col_name, title, fname, label_text):
    lower = df_sub[col_name].quantile(0.005)
    upper = df_sub[col_name].quantile(0.995)
    df_filtered = df_sub[(df_sub[col_name] >= lower) & (df_sub[col_name] <= upper)]

    plt.figure(figsize=(16, 7))
    sns.histplot(df_filtered[df_filtered['label'] == 0][col_name], color='tab:blue', label='Human', kde=True, stat='density', bins=30, alpha=0.7)
    sns.histplot(df_filtered[df_filtered['label'] == 1][col_name], color='tab:red', label='LLM', kde=True, stat='density', bins=30, alpha=0.7)
    
    if col_name in threshold_dict:
        if 'sweep' in threshold_dict[col_name]:
            thr_sweep = threshold_dict[col_name]['sweep']
            plt.axvline(thr_sweep, color='forestgreen', linestyle='--', label=f'Sweep={thr_sweep:.3f}')
        if 'kde' in threshold_dict[col_name]:
            thr_kde = threshold_dict[col_name]['kde']
            plt.axvline(thr_kde, color='orange', linestyle='--', label=f'KDE={thr_kde:.3f}')
    
    plt.title(title)
    plt.xlabel(label_text)
    plt.ylabel("Density")
    plt.legend(loc='upper right', framealpha=0.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


# KDE 기반 threshold 함수
from scipy.optimize import fsolve 

def find_threshold_by_kde(scores_human, scores_llm):
    min_val = min(scores_human.min(), scores_llm.min())
    max_val = max(scores_human.max(), scores_llm.max())
    x_range = np.linspace(min_val - 3 * scores_human.std(), max_val + 3 * scores_llm.std(), 1000) 

    kde_human = gaussian_kde(scores_human)
    kde_llm = gaussian_kde(scores_llm)

    pdf_human = kde_human(x_range)
    pdf_llm = kde_llm(x_range)

    diff_func = lambda x: kde_human(x) - kde_llm(x)
    cross_indices = np.where(np.diff(np.sign(pdf_human - pdf_llm)))[0]

    if len(cross_indices) > 0:
        initial_guess = x_range[cross_indices[0]] 
        try:
            threshold = fsolve(diff_func, initial_guess)[0]
            if min_val <= threshold <= max_val:
                return threshold
            else: 
                return (np.median(scores_human) + np.median(scores_llm)) / 2
        except Exception:
            return (np.median(scores_human) + np.median(scores_llm)) / 2
    else:
        return (np.median(scores_human) + np.median(scores_llm)) / 2


# threshold sweep 및 KDE 함수
def evaluate_and_report(y_true, scores, threshold, method_name, score_name, save_prefix):
    pred = (scores > threshold).astype(int)
    f1 = f1_score(y_true, pred)
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, scores) 

    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Human', 'LLM'], yticklabels=['Human', 'LLM'])
    plt.title(f"[{score_name} - {method_name}]\nThreshold={threshold:.3f}\nF1={f1:.3f}, ACC={acc:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{save_prefix}_{method_name}_confmat_testset.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[{score_name} - {method_name}]")
    print(f" Threshold: {threshold:.3f}")
    print(f" Test Set → F1={f1:.3f}, Accuracy={acc:.3f}, AUC={auc:.3f}")
    print(classification_report(y_true, pred, digits=3, zero_division=0))


def process_score_column(train_df, test_df, score_col, score_name, save_prefix):
    global threshold_dict

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    train_scores = train_df[score_col].values
    test_scores = test_df[score_col].values
    
    # 1. Sweep 기반 임계값 탐색
    best_f1_sweep, best_thr_sweep = -1, None
    thresholds_sweep = np.linspace(train_scores.min(), train_scores.max(), 50)
    for thr in thresholds_sweep:
        pred = (train_scores > thr).astype(int)
        f1 = f1_score(y_train, pred)
        if f1 > best_f1_sweep:
            best_f1_sweep = f1
            best_thr_sweep = thr
    
    # 2. KDE 기반 임계값 탐색
    scores_human_train = train_df[train_df['label'] == 0][score_col].values
    scores_llm_train = train_df[train_df['label'] == 1][score_col].values
    best_thr_kde = find_threshold_by_kde(scores_human_train, scores_llm_train)

    threshold_dict[score_col] = {
        'sweep': best_thr_sweep,
        'kde': best_thr_kde
    }

    print(f"\n{'='*50}\nProcessing: {score_name}\n{'='*50}")

    evaluate_and_report(y_test, test_scores, best_thr_sweep, "Sweep", score_name, f"{save_prefix}_sweep")
    evaluate_and_report(y_test, test_scores, best_thr_kde, "KDE", score_name, f"{save_prefix}_kde")


def main():
    df_all = pd.concat([df_train, df_test])

    process_score_column(df_train, df_test, 'tocsin_z', "TOCSIN z-score", "tocsin_zscore")
    process_score_column(df_train, df_test, 'simllm_z', "SimLLM z-score", "simllm_zscore")
    process_score_column(df_train, df_test, 'ensemble_z_weighted', "Weighted z-score", "weighted_zscore")
    process_score_column(df_train, df_test, 'ensemble_sigmoid', "Sigmoid Ensemble", "sigmoid_ensemble")

    draw_hist_right(df_all, 'tocsin_z', 'TOCSIN z-score Distribution', 'tocsin_zscore_by_label.png', 'TOCSIN z-score')
    draw_hist(df_all, 'simllm_z', 'SimLLM z-score Distribution', 'simllm_zscore_by_label.png', 'SimLLM z-score')
    draw_hist(df_all, 'ensemble_z_weighted', 'Weighted Ensemble z-score Distribution', 'ensemble_zscore_weighted_by_label.png', 'Weighted Ensemble z-score')
    draw_hist(df_all, 'ensemble_sigmoid', 'Sigmoid Ensemble z-score Distribution', 'ensemble_sigmoid_by_label.png', 'Sigmoid Ensemble Score')

    print("Raw 점수 기반 실험 완료! 결과 저장됨:", SAVE_DIR)

if __name__ == "__main__":
    main()