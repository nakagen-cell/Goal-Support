from __future__ import annotations

import json
import os
import re
from typing import Dict, Tuple, Any, List

from .models import Directiveness, ChoiceFraming

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
# Two-stage generation: Content Planner -> Verbalizer
# =========================================================

CHOICE_PHRASE = "無理のない範囲で、どれを選ぶかはあなたが決めて構いません。"


# === STRICT DIFFERENTIATION RULES (guarantee perceptual separation) ===
# These rules are used as additional deviation checks during rerendering.
MANDATORY_HIGH_COMMAND_PHRASES = ["してください", "やろう", "始めよう", "進めよう", "やってみよう"]  # varied command cues in HIGH  # at least one required in HIGH
MANDATORY_LOW_SUGGESTION_PATTERNS = [
    "してみる",
    "するのはどうでしょう",
    "するのも一つの手です",
]
IMPLICIT_CHOICE_LEAK_PATTERNS = [
    # "自由度提示（主体メタ）"に相当する明示的な表現のみを検出する。
    # 内容語彙として自然に出やすい語（例：『調整』単体）は含めない。
    re.compile(r"(あなた|自分).*(選|決)"),
    re.compile(r"(自由に|好きに).*(選|決|調整)"),
    re.compile(r"(どれでも|どちらでも|いずれでも).*?(いい|OK|大丈夫)?"),
    re.compile(r"(選んで|選択して).*(ください|OK|大丈夫)?"),
    re.compile(r"無理のない範囲で.*(調整|決め|選)"),
]


def _strict_style_flags(output_text: str, directiveness: str, choice_framing: str) -> list[str]:
    flags: list[str] = []
    if directiveness == "HIGH" and not any(p in output_text for p in MANDATORY_HIGH_COMMAND_PHRASES):
        flags.append("STRICT_HIGH_MISSING_COMMAND_PHRASE")
    if directiveness == "LOW" and not any(p in output_text for p in MANDATORY_LOW_SUGGESTION_PATTERNS):
        flags.append("STRICT_LOW_MISSING_SUGGESTION_PATTERN")
    if choice_framing == "ABSENT":
        if any(r.search(output_text) for r in IMPLICIT_CHOICE_LEAK_PATTERNS):
            flags.append("STRICT_ABSENT_CHOICE_LEAK")
    return flags
# === END STRICT DIFFERENTIATION RULES ===

# Simple deviation detection rules
_IMPERATIVE_PATTERNS = [
    re.compile(r"(必ず|絶対に|今すぐ)"),
    re.compile(r"(しなさい|しろ|せよ)"),
    re.compile(r"(しなければならない|しなければなりません|〜?ねばならない)"),
]


def _detect_deviation(
    *,
    text: str,
    directiveness: Directiveness,
    choice_framing: ChoiceFraming,
) -> List[str]:
    flags: List[str] = []

    if directiveness == Directiveness.LOW:
        for pat in _IMPERATIVE_PATTERNS:
            if pat.search(text):
                flags.append("FORBIDDEN_IMPERATIVE")
                break

    if choice_framing == ChoiceFraming.PRESENT:
        if CHOICE_PHRASE not in text:
            flags.append("MISSING_CHOICE_PHRASE")
    else:
        # ABSENT
        if CHOICE_PHRASE in text:
            flags.append("FORBIDDEN_CHOICE_PHRASE")

        # also block common "you can choose" meta lines (best-effort)
        if re.search(r"(どれでも(よい|OK)|あなたが決めて|選ばないという選択|自由に選)", text):
            flags.append("CHOICE_META_LEAK")

    return flags



def _detect_content_mismatch(*, plan: Dict[str, Any], text: str) -> List[str]:
    """Detect whether the rendered text faithfully contains the fixed content plan.

    We enforce *string-level* inclusion (no paraphrase) for:
      - option ids (A/B/C)
      - action strings
      - duration_min numbers (as '<n>分')
      - every step string

    Reason lines may be summarized, so we do not require exact 'reason' matches.
    """
    flags: List[str] = []
    options = plan.get("options") or []
    # Defensive: ensure list[dict]
    if not isinstance(options, list):
        return ["CONTENT_PLAN_OPTIONS_NOT_LIST"]

    for opt in options:
        if not isinstance(opt, dict):
            continue
        oid = str(opt.get("id") or "").strip()
        action = str(opt.get("action") or "").strip()
        duration = opt.get("duration_min")
        steps = opt.get("steps") or []

        if oid:
            if f"{oid})" not in text:
                flags.append(f"CONTENT_MISSING_OPTION_{oid}")
        if action and action not in text:
            flags.append(f"CONTENT_MISSING_ACTION_{oid or 'UNK'}")
        try:
            d_int = int(duration)
            if f"{d_int}分" not in text:
                flags.append(f"CONTENT_MISSING_DURATION_{oid or 'UNK'}")
        except Exception:
            flags.append(f"CONTENT_BAD_DURATION_{oid or 'UNK'}")

        if isinstance(steps, list):
            for i, st in enumerate(steps):
                st_s = str(st).strip()
                if st_s and st_s not in text:
                    flags.append(f"CONTENT_MISSING_STEP_{oid or 'UNK'}_{i+1}")
        else:
            flags.append(f"CONTENT_STEPS_NOT_LIST_{oid or 'UNK'}")

    return flags

def _integrity_metrics(plan: Dict[str, Any], text: str) -> Dict[str, int]:
    opts = plan.get("options") or []
    num_options = len(opts)
    num_steps_total = 0
    for o in opts:
        steps = o.get("steps") or []
        num_steps_total += len(steps)
    return {
        "num_options": int(num_options),
        "num_steps_total": int(num_steps_total),
        "char_count": int(len(text)),
    }


def generate_content_plan(*, goal: str, context_text: str) -> Tuple[str, Dict[str, Any]]:
    """Stage A: generate a structured content plan (JSON) from goal + context.

    Returns (prompt_json_str, plan_dict)
    """
    cli = _ensure_client()

    system_prompt = (
        "あなたは行動支援メッセージの『内容プランナー』です。\n"
        "目的：Goal と Context から、提案内容のみを JSON で設計してください。\n"
        "制約：\n"
        "1) 出力は JSON のみ（説明文や前置きは禁止）。\n"
        "2) options は必ず 3 つ（A/B/C）で固定。\n"
        "3) 各 option は action(短い行動名), duration_min(整数), steps(2〜3個), reason(1文) を含む。\n"
        "4) 表現（命令形/丁寧語/自由度提示）は一切書かない。『何をするか』だけを書く。\n"
    )
    user_payload = {"goal": goal, "context": context_text}
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "次の入力に基づいて content plan を JSON で出力してください。\n"
                + json.dumps(user_payload, ensure_ascii=False)
            ),
        },
    ]

    resp = cli.chat.completions.create(
        model=os.getenv("GOAL_SUPPORT_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: wrap into minimal plan
        plan = {
            "goal": goal,
            "context": context_text,
            "options": [
                {
                    "id": "A",
                    "action": "目標に向けた短い作業をする",
                    "duration_min": 10,
                    "steps": ["開始準備をする", "10分だけ取り組む"],
                    "reason": "短時間でも進捗につながるため",
                },
                {
                    "id": "B",
                    "action": "作業量を小さくして続ける",
                    "duration_min": 5,
                    "steps": ["一番小さい単位を決める", "5分だけやる"],
                    "reason": "負担を下げて継続しやすくするため",
                },
                {
                    "id": "C",
                    "action": "次にやることを決めて終える",
                    "duration_min": 3,
                    "steps": ["次の一手を書く", "必要な道具を用意する"],
                    "reason": "再開コストを下げるため",
                },
            ],
        }

    # Normalize / enforce A,B,C
    options_raw = plan.get("options") or []

    # --- normalize options to list[dict] (LLM may return dict instead of list) ---
    options = []
    if isinstance(options_raw, list):
        options = [o if isinstance(o, dict) else {} for o in options_raw]
    elif isinstance(options_raw, dict):
        def _sort_key(k):
            try:
                return (0, int(k))
            except Exception:
                return (1, str(k))
        for k in sorted(options_raw.keys(), key=_sort_key):
            v = options_raw[k]
            options.append(v if isinstance(v, dict) else {})
    # --- end normalize ---

    fixed = []
    for i, k in enumerate(["A", "B", "C"]):
        src = options[i] if i < len(options) else {}
        fixed.append(
            {
                "id": k,
                "action": str(src.get("action") or "短い作業をする"),
                "duration_min": int(src.get("duration_min") or 10),
                "steps": list(src.get("steps") or ["開始する", "短時間だけ進める"])[:3],
                "reason": str(src.get("reason") or "負担を増やさず進めるため"),
            }
        )
    plan["goal"] = goal
    plan["context"] = context_text
    plan["options"] = fixed

    prompt_json = json.dumps(messages, ensure_ascii=False)
    return prompt_json, plan


def _verbalizer_system_prompt(d: Directiveness, c: ChoiceFraming) -> str:
    # We keep UI/structure fixed; only expression parameters vary.
    base = (
        "あなたは行動支援メッセージの『表現レンダラー』です。\n"
        "入力の content_plan(JSON) を、指定条件に従って日本語の支援メッセージ本文として出力してください。\n"
        "重要：内容（action/duration/steps/reason）は一切変えない。言い回しだけ変える。\n"
        "出力フォーマット（全条件で固定）：\n"
        "1) 最初に 1〜2 文の導入。\n"
        "2) その後、A)/B)/C) の3行（1行=1案）。各行に duration(分) と steps を『→』で含める。\n"
        "   例：A) 行動（10分）：手順1→手順2→手順3\n"
        "3) 最後に reason を 1 行でまとめる（3案の理由を短くまとめてよい）。\n"
    )
    if d == Directiveness.HIGH:
        base += (
            "命令性：HIGH。導入文と A)/B)/C) は、断定的・命令的でもよい。回りくどくしない。\n"
        )
    else:
        base += (
            "命令性：LOW。命令形（〜しなさい/〜しろ/必ず/絶対に/今すぐ 等）を使わず、提案表現にする。\n"
        )
    if c == ChoiceFraming.PRESENT:
        base += (
            f"自由度提示：PRESENT。本文のどこかに次の定型句を『そのまま一字一句』必ず1回入れる：{CHOICE_PHRASE}\n"
            "定型句以外で『選べる』『自由に』などのメタ表現を増やしすぎない。\n"
        )
    else:
        base += (
            "自由度提示：ABSENT。主体の所在（あなたが決めて等）や『どれでもよい』『選ばないという選択』などのメタ表現は禁止。\n"
        )
    return base


def render_instruction(
    *,
    content_plan: Dict[str, Any],
    directiveness: Directiveness,
    choice_framing: ChoiceFraming,
    max_rerender: int = 2,
) -> Tuple[str, str, List[str], int]:
    """Stage B: render the fixed content plan into text, applying (d, c).

    Returns (prompt_json_str, output_text, deviation_flags, rerender_count)
    """
    cli = _ensure_client()

    sys = _verbalizer_system_prompt(directiveness, choice_framing)
    messages = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": json.dumps({"content_plan": content_plan}, ensure_ascii=False),
        },
    ]

    rerender_count = 0
    last_text = ""
    last_flags: List[str] = []

    for attempt in range(max_rerender + 1):
        resp = cli.chat.completions.create(
            model=os.getenv("GOAL_SUPPORT_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        flags = _detect_deviation(
            text=text,
            directiveness=directiveness,
            choice_framing=choice_framing,
        )
        # Content fidelity checks (do not allow the verbalizer to drop/alter plan fields)
        flags.extend(_detect_content_mismatch(plan=content_plan, text=text))
        # Additional strict style separation checks (must differ in impression) (must differ in impression)
        flags.extend(_strict_style_flags(text, directiveness.value, choice_framing.value))

        last_text, last_flags = text, flags
        if not flags:
            break

        # Ask for a strict correction without changing content
        rerender_count += 1
        messages.append(
            {
                "role": "user",
                "content": (
                    "条件に反する表現が混入しています。内容は絶対に変えず、表現だけを修正して再出力してください。\n"
                    f"違反フラグ: {flags}"
                ),
            }
        )

    prompt_json = json.dumps(messages, ensure_ascii=False)
    return prompt_json, last_text, last_flags, rerender_count


def generate_instruction(
    *,
    goal: str,
    context_text: str = "",
    directiveness: Directiveness,
    choice_framing: ChoiceFraming,
    fixed_content_plan_json: str | None = None,
) -> Tuple[str, str, str, Dict[str, int], List[str], int]:
    """High-level API: content plan -> verbalize, with integrity/deviation info.

    Returns:
      - llm_prompt_json_str (planner+verbalizer prompts)
      - llm_output_text
      - content_plan_json_str
      - integrity_metrics dict
      - deviation_flags list
      - rerender_count
    """
    planner_prompt = None
    if fixed_content_plan_json:
        try:
            plan = json.loads(fixed_content_plan_json)
        except Exception:
            # If the stored plan is corrupted, fall back to re-planning.
            planner_prompt, plan = generate_content_plan(goal=goal, context_text=context_text)
    else:
        planner_prompt, plan = generate_content_plan(goal=goal, context_text=context_text)
    verbal_prompt, out, flags, rerender_count = render_instruction(
        content_plan=plan,
        directiveness=directiveness,
        choice_framing=choice_framing,
    )

    combined_prompt = json.dumps(
        {
            "planner_prompt": (json.loads(planner_prompt) if planner_prompt else "<FIXED_CONTENT_PLAN>"),
            "verbalizer_prompt": json.loads(verbal_prompt),
            "directiveness": directiveness.value,
            "choice_framing": choice_framing.value,
        },
        ensure_ascii=False,
    )
    plan_json = json.dumps(plan, ensure_ascii=False)
    integrity = _integrity_metrics(plan, out)

    return combined_prompt, out, plan_json, integrity, flags, rerender_count
