# Single-sentence pseudo pairing for TOCSIN + SimLLM:
# 입력 문장 x -> x_co(continuation), x_re_h(x의 regeneration), x_re_g(x_co의 regeneration)

# - 모델/프롬프트/temperature 등은 기존 코드 스타일을 최대한 유지
# - whitespace 기반 20토큰 접두(prefix_20tok) 추출 유틸 포함
# - gpt-3.5-turbo 사용 (교체 시 model 변수만 바꾸면 됨)

import os
import time
import math
from typing import Dict
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")  # 환경변수 또는 직접 입력
MODEL_ID = os.environ.get("MODEL_ID", "gpt-3.5-turbo")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
MAX_TOKENS_CONT = int(os.environ.get("MAX_TOKENS_CONT", "256"))
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "1.2"))  # 속도 조절 (요청 간격)
MAX_RETRIES = 3

client = OpenAI(api_key=OPENAI_API_KEY)

def _retry_call(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            sleep = 1.5 ** attempt
            print(f"[warn] OpenAI 호출 실패({attempt}/{MAX_RETRIES}): {e} -> {sleep:.1f}s 후 재시도")
            time.sleep(sleep)

def _chat_once(system_prompt: str, user_prompt: str) -> str:
    resp = _retry_call(
        client.chat.completions.create,
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt} if system_prompt else {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_CONT
    )
    out = resp.choices[0].message.content.strip()
    out = out.replace('\"', '').strip()
    return out

def prefix_20tok(text: str, k: int = 20) -> str:
    """whitespace 기준 k 토큰 접두 추출 (문장 길이가 짧으면 절반 길이 보정)"""
    toks = text.strip().split()
    if len(toks) <= k:
        k_eff = max(1, len(toks) // 2)
        return " ".join(toks[:k_eff]).strip()
    return " ".join(toks[:k]).strip()

def build_prompt_cont(prefix: str, target_len_chars: int) -> str:
    return (
        f"다음 뉴스 기사 문장의 앞부분에 이어서 전체 문장을 자연스럽고 완결성 있게 작성하세요.\n"
        f"전체 문장의 길이는 약 {target_len_chars}자 내외가 되도록 해주세요. "
        f"아래는 문장의 시작 부분입니다: '{prefix}'"
    )

def build_prompt_regen(text: str) -> str:
    return (
        "다음 문장을 의미는 그대로 유지하되, 문법적으로 더 유창하고 자연스럽게 다시 써 주세요.\n"
        "단, 문장의 말투(예: '~한다.', '~이다.' 등)는 바꾸지 말고, 원래 문장의 어미 형태를 그대로 유지하세요.\n"
        "또한, 불필요한 설명 없이 수정된 문장만 한 줄로 출력하세요.\n\n"
        f"문장:\n{text}"
    )

def generate_pairset_for_single_x(x: str) -> Dict[str, str]:
    """
    입력: 단일 문장 x (정체 미상: human/LLM)
    출력: dict {
        'x'        : 원문
        'x_co'     : continuation (TOCSIN용)
        'x_re_h'   : x의 regeneration (SimLLM용 human-side)
        'x_re_g'   : x_co의 regeneration (SimLLM용 gpt-side)
        'prefix_20tok': 사용된 접두
    }
    """
    x = x.strip()

    # 1) continuation: x의 접두(prefix_20tok)만 주고 자연스러운 이어쓰기 생성
    pref = prefix_20tok(x, k=20)
    target_len = max(50, int(len(x) * 0.95)) 
    cont_prompt = build_prompt_cont(pref, target_len_chars=target_len)
    x_co = _chat_once(system_prompt="", user_prompt=cont_prompt)
    x_co_full = (pref + " " + x_co).strip()

    time.sleep(SLEEP_SEC)

    # 2) regeneration(h): x를 그대로 더 유창하게 (어미 유지)
    regen_prompt_h = build_prompt_regen(x)
    x_re_h = _chat_once(system_prompt="당신은 한국어 문장을 교정하는 전문가입니다.", user_prompt=regen_prompt_h)

    time.sleep(SLEEP_SEC)

    # 3) regeneration(g): x_co_full을 다시 유창하게 (어미 유지)
    regen_prompt_g = build_prompt_regen(x_co_full)
    x_re_g = _chat_once(system_prompt="당신은 한국어 문장을 교정하는 전문가입니다.", user_prompt=regen_prompt_g)

    return {
        "x": x,
        "x_co": x_co_full,
        "x_re_h": x_re_h,
        "x_re_g": x_re_g,
        "prefix_20tok": pref
    }


if __name__ == "__main__":
    
    import sys, json
    if sys.stdin.isatty():
        sample = "정부는 내년도 경제정책 방향을 발표하며 물가 안정과 내수 회복을 최우선 과제로 제시했다."
        print("[demo] 입력 x:", sample)
        result = generate_pairset_for_single_x(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        x_in = sys.stdin.read().strip()
        result = generate_pairset_for_single_x(x_in)
        print(json.dumps(result, ensure_ascii=False, indent=2))
