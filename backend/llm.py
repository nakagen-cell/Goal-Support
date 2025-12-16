from __future__ import annotations

import json
import os
import re
from typing import Dict, Tuple

from .models import Style

# =========================================================
# OpenAI client
# =========================================================
_client = None


def _ensure_client():
    """Lazily construct an OpenAI client instance."""
    global _client
    if _client is not None:
        return _client
    from openai import OpenAI  # openai>=1.x

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    _client = OpenAI(api_key=api_key)
    return _client


# =========================================================
# Style-specific system prompts
# =========================================================

STYLE_SYSTEM_PROMPTS: Dict[Style, str] = {
    # 指示スタイル：形式は選択肢付きだが、実質的には「A) をやれ」と強く指示する
    Style.DIRECTIVE: (
        "あなたは、ユーザーに対して行動を明確かつ強く指示するコーチです。\n"
        "・最初に 1〜2 文の導入文を書き、そのあとに選択肢形式で行動案を提示してください。\n"
        "・「A)」には、ユーザーが必ず実行すべきメインの行動を 1 つだけ書いてください。\n"
        "・必要であれば B) や C) に補助的なサブ行動（例：確認や準備）を書いても構いませんが、A) 以外はあくまで付随的な位置づけにしてください。\n"
        "・各選択肢は「〜しなさい」「〜しろ」「〜しなければならない」「必ず〜しなさい」などの命令形で書き、特に A) の行動は強く指示してください。\n"
        "・「どれを選んでもよい」「選ばないという選択もある」といった文は書かないでください。A) を実行することを前提とした文脈にしてください。\n"
        "・出力は日本語で行い、上からの一方向の指示として読める文体にしてください。\n"
    ),

    # 提案スタイル：複数の選択肢を明示し、ユーザーが自分で選べる形式
    Style.SUGGESTIVE: (
        "あなたは、ユーザーの自律性を尊重する行動支援アシスタントです。\n"
        "・最初に 1〜2 文の導入文を書き、そのあとに選択肢を提示してください。\n"
        "・必ず 3〜4 個の行動選択肢を A) / B) / C) / D) の形式で、1 行につき 1 つずつ提示してください。\n"
        "  例:『A) 今日は◯分だけ取り組んでみる』『B) 時間を短くして軽めにやる』のように書いてください。\n"
        "・各選択肢は「〜してみる」「〜してみませんか」「〜という選択肢もあります」のような穏やかな提案表現にしてください。\n"
        "・命令表現（〜しなさい、〜しろ、〜しなければならない、必ず〜、今すぐ〜、絶対に〜 など）は一切使わないでください。\n"
        "・本文のどこかで「どれを選んでも構いません」「選ばないという選択もあります」「あなたが決めて大丈夫です」といった文を必ず 1 文以上入れてください。\n"
        "・出力は日本語で行い、穏やかで押しつけにならない語調にしてください。\n"
    ),

    # 協働スタイル：質問と共感を通じて一緒に選択肢を検討する形式
    Style.COLLABORATIVE: (
        "あなたは、ユーザーと一緒に計画を考える協働パートナーです。\n"
        "・最初の 1〜2 文は、必ず質問文にして、ユーザーの気分や体調、ペース、優先したいことをたずねてください。\n"
        "・その後で、A) / B) / C) 形式で 2〜3 個の行動案を提示してください。各案は「もしよければ〜してみましょう」「〜というやり方も一緒に考えられます」のように、共に考える姿勢で書いてください。\n"
        "・命令表現は使わないでください。代わりに、「無理のない範囲で」「負担にならない形で」など、ユーザーの状態に寄り添うフレーズを含めてください。\n"
        "・本文のどこかで、「一緒に」「いっしょに」「相談しながら」「あなたの気持ちを大事にしながら」といった協働・共感を示す語を 1 語以上使ってください。\n"
        "・最後に、「どの案を選んでも構いません」「あなたのペースで一緒に考えていきましょう」など、共同で選ぶ雰囲気を伝える一文を入れてください。\n"
        "・出力は日本語で行い、丁寧で共感的な文体にし、上から指示する印象を与えないようにしてください。\n"
    ),
}

# =========================================================
# Simple post-hoc filter for coercive phrases
# =========================================================

# Mapping from coercive phrase to a softer alternative.
_REPLACE_MAP = [
    (re.compile(r"必ず"), "できれば"),
    (re.compile(r"しなければならない"), "したほうが良いかもしれません"),
    (re.compile(r"しなければなりません"), "するのも一つの選択肢です"),
    (re.compile(r"今すぐ"), "タイミングが合うときに"),
    (re.compile(r"絶対に"), "できる範囲で"),
    (re.compile(r"しなさい"), "してみるのも良いかもしれません"),
]


def _apply_non_coercive_filter(style: Style, text: str) -> str:
    """Apply a simple rule-based filter to reduce coercive expressions.

    提案スタイルおよび協働スタイルに対してのみ適用し，
    LLM の確率的逸脱による統制的表現の混入を軽減する。
    """
    if style not in (Style.SUGGESTIVE, Style.COLLABORATIVE):
        return text

    filtered = text
    for pattern, replacement in _REPLACE_MAP:
        filtered = pattern.sub(replacement, filtered)
    return filtered


# =========================================================
# Instruction generation API
# =========================================================

def generate_instruction(
    *,
    goal: str,
    style: Style,
    context_text: str = "",
) -> Tuple[str, str]:
    """Generate one instruction message given goal, style, and context.

    目標 (Goal)、当日の状況・履歴 (Context)、文体条件 (Style) を入力として，
    1 回分の支援メッセージを生成する。戻り値は

    - llm_prompt_json_str: LLM に渡したメッセージ列（JSON 文字列）
    - llm_output_text: ユーザーに提示するメッセージ本文

    のタプルである。
    """
    cli = _ensure_client()
    if style not in STYLE_SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported style: {style}")

    system_prompt = STYLE_SYSTEM_PROMPTS[style]

    user_payload = {
        "goal": goal,
        "context": context_text,
        "instruction_style": style.value,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "以下の情報を踏まえて、1回分の支援メッセージを生成してください。\n"
                "出力は、ユーザーに直接表示する本文のみとし、前置きやメタな説明は書かないでください。\n"
                + json.dumps(user_payload, ensure_ascii=False)
            ),
        },
    ]

    resp = cli.chat.completions.create(
        model=os.getenv("GOAL_SUPPORT_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.4,
    )
    raw = resp.choices[0].message.content or ""
    filtered = _apply_non_coercive_filter(style, raw.strip())

    prompt_json = json.dumps(messages, ensure_ascii=False)
    return prompt_json, filtered
