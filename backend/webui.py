from __future__ import annotations

import os

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
from typing import Optional
from .repo import (
    SessionLocal,
    create_anonymous_user,
    get_or_create_singleton_user,
    create_session,
    get_session,
    get_current_condition,
    within_is_complete,
    advance_condition,
    log_turn,
    update_evaluation,
    set_fixed_task_plan,
)
from .models import Directiveness, ChoiceFraming, TurnLog, ExperimentSession
from .llm import generate_instruction, generate_content_plan
import html
import random

router = APIRouter()


HTMX = '<script src="https://unpkg.com/htmx.org@1.9.12"></script>'
TAILWIND = '<script src="https://cdn.tailwindcss.com"></script>'


def _wrap_page(body: str) -> str:
    # グローバルな処理中インジケータ（LOAD 表示）を追加
    loading = """
    <div id="global-loading"
         class="htmx-indicator fixed inset-x-0 top-0 flex justify-center mt-2 z-50">
      <div class="px-4 py-1 rounded-full bg-slate-900/80 text-white text-xs font-mono tracking-widest shadow-lg">
        LOAD...
      </div>
    </div>
    """
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Instruction Expression Control UI</title>
  {TAILWIND}
  {HTMX}
  <style>
    /* htmx インジケータをデフォルト非表示に */
    .htmx-indicator {{ opacity: 0; pointer-events: none; transition: opacity 0.15s ease; }}
    .htmx-request .htmx-indicator {{ opacity: 1; pointer-events: auto; }}
  </style>
</head>
<body class="bg-slate-50 text-slate-900">
  {loading}
  <div class="max-w-4xl mx-auto p-4 space-y-4">
    <h1 class="text-2xl font-bold">LLM 表現要素（命令性×自由度提示）制御 UI</h1>
    <p class="text-sm text-slate-600">
      命令性（HIGH/LOW）×自由度提示（PRESENT/ABSENT）を切り替えて、支援メッセージと主観評価を収集するための簡易 Web UI です。
      （サーバー処理中は画面上部に <span class="font-mono">LOAD...</span> が表示されます）
    </p>
    <div id="alerts"></div>
    {body}
  </div>

  <script>
    (function() {{
      function scrollConvoToBottom() {{
        var box = document.querySelector("#conversation .js-convo-box");
        if (!box) return;
        box.scrollTop = box.scrollHeight;
      }}

      // Initial load
      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", scrollConvoToBottom);
      }} else {{
        scrollConvoToBottom();
      }}

      // After HTMX updates
      document.body.addEventListener("htmx:afterSwap", function(evt) {{
        // Our UI swaps #conversation outerHTML, so always try to scroll after swap
        scrollConvoToBottom();
      }});

      // Some browsers/layouts settle after a tick; this helps ensure bottom alignment.
      document.body.addEventListener("htmx:afterSettle", function(evt) {{
        scrollConvoToBottom();
      }});
    }})();
  </script>

</body>
</html>"""


def _label_directiveness(d: Directiveness) -> str:
    return "命令性 HIGH" if d == Directiveness.HIGH else "命令性 LOW"


def _label_choice_framing(c: ChoiceFraming) -> str:
    return "自由度提示 PRESENT" if c == ChoiceFraming.PRESENT else "自由度提示 ABSENT"


def _render_conversation(es: Optional[ExperimentSession]) -> str:
    """Render conversation + evaluation area for a given session."""
    if es is None:
        return """
        <div id="conversation" class="space-y-3">
          <div class="p-3 bg-white rounded-xl shadow text-sm text-slate-500">
            まだセッションが開始されていません。上のフォームから開始してください。
          </div>
        </div>
        """

    session_id = es.id
    with SessionLocal() as s:
        logs = (
            s.query(TurnLog)
            .filter_by(session_id=session_id)
            .order_by(TurnLog.turn_index.asc())
            .all()
        )

    d_curr, c_curr, cond_idx, cond_total = get_current_condition(es)

    # Within-subject sessions are considered "completed" when condition_index >= total.
    # Used to stop further generation and show a completion notice.
    completed = within_is_complete(es)

    is_within = (getattr(es, "design_mode", "BETWEEN") or "BETWEEN").upper() == "WITHIN"

    header = f"""
    <div class=\"p-3 bg-slate-100 rounded-lg text-xs text-slate-700\">
      <div>セッションID: <span class=\"font-mono\">{session_id}</span></div>
      <div>条件: <span class=\"font-semibold\">{html.escape(_label_directiveness(d_curr))}</span>
           / <span class=\"font-semibold\">{html.escape(_label_choice_framing(c_curr))}</span></div>
      <div>条件進捗: <span class=\"font-mono\">{cond_idx + 1}</span> / <span class=\"font-mono\">{cond_total}</span></div>
      <div>ゴール: {html.escape(es.goal)}</div>
      <div>コンテキスト: {html.escape(getattr(es, 'initial_context', '') or '')}</div>
    </div>
    """


    bubbles = []
    for log in logs:

        # System/LLM message (show after the user message)
        bubbles.append(
            f'<div class="flex justify-start">'
            f'<div class="max-w-[80%] p-3 my-1 rounded-xl bg-white shadow text-sm whitespace-pre-line">'
            f'<div class="text-xs text-slate-400 mb-1">LLM</div>'
            f'{html.escape(log.llm_output)}</div></div>'
        )

    convo_html = "".join(bubbles) or """
      <div class="text-sm text-slate-500">
        まだメッセージはありません。「次のメッセージを生成」から開始してください。
      </div>
    """

    # last turn for evaluation form
    last_turn_id = logs[-1].id if logs else ""

    # Unified form: evaluation (for the latest LLM message) + next action (generate next message)
    # This reduces button-order confusion and guarantees evaluation is tied to the immediately previous turn.
    last_turn_id = logs[-1].id if logs else ""

    if completed:
        return f"""
    <div id=\"conversation\" class=\"space-y-3\">
      {header}
      <div class=\"p-3 bg-emerald-50 rounded-xl border text-sm text-emerald-900\">
        条件の提示と評価が完了しました。ご協力ありがとうございました。
      </div>
    </div>
    """
    # --- unified: we do NOT collect participant replies nor action choices ---
    # Keep DB columns for compatibility, but never show or require these inputs in the UI.
    ADVANCE_INPUT_BLOCK = '<input type="hidden" name="action_choice" value="EXECUTE" />'

    advance_form = f"""
    <form
      hx-post="/ui/advance_turn"
      hx-target="#conversation"
      hx-swap="outerHTML"
      hx-indicator="#global-loading"
      class="mt-3 space-y-3"
    >
      <input type="hidden" name="session_id" value="{session_id}" />
      <input type="hidden" name="turn_id" value="{last_turn_id}" />

      {ADVANCE_INPUT_BLOCK}

      <div class="text-sm font-semibold text-slate-700">直前のLLMメッセージに対する評価</div>
      <div class="text-xs text-slate-500 mb-2">※ 各項目は「1 = 該当しない」「5 = 該当する」で回答してください。</div>

      <div class="grid grid-cols-2 gap-3 text-xs items-start">
        <div class="p-3 bg-slate-50 rounded-xl border space-y-2">
          <div class="font-semibold text-slate-700">自律性（1〜5, 3項目）</div>
          <div class="space-y-2">
            <label class="block">
              <div class="text-slate-600">Q1. この提案は「自分で選んでいる」と感じた</div>
              <input type="number" name="autonomy_i1" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q2. この提案に従うかどうかは「自分次第」だと感じた</div>
              <input type="number" name="autonomy_i2" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q3. この提案は「自分の意思」を尊重していると感じた</div>
              <input type="number" name="autonomy_i3" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
          </div>
        </div>

        <div class="p-3 bg-slate-50 rounded-xl border space-y-2">
          <div class="font-semibold text-slate-700">やらされ感（統制知覚）（1〜5, 4項目）</div>
          <div class="space-y-2">
            <label class="block">
              <div class="text-slate-600">Q1. 指示されている／やらされている感じがした</div>
              <input type="number" name="coercion_i1" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q2. 圧力や強制を感じた</div>
              <input type="number" name="coercion_i2" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q3. 断れない雰囲気／従うべき雰囲気を感じた</div>
              <input type="number" name="coercion_i3" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q4. 自分の自由が制限されたように感じた</div>
              <input type="number" name="coercion_i4" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
          </div>
        </div>

        <div class="p-3 bg-slate-50 rounded-xl border space-y-2">
          <div class="font-semibold text-slate-700">操作チェック：命令性の強さ（1〜5, 2項目）</div>
          <div class="space-y-2">
            <label class="block">
              <div class="text-slate-600">Q1. 命令・指示の口調が強いと感じた</div>
              <input type="number" name="pd_i1" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q2. 「〜すべき／〜しなさい」に近いニュアンスを感じた</div>
              <input type="number" name="pd_i2" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
          </div>
        </div>

        <div class="p-3 bg-slate-50 rounded-xl border space-y-2">
          <div class="font-semibold text-slate-700">操作チェック：自由度提示（1〜5, 2項目）</div>
          <div class="space-y-2">
            <label class="block">
              <div class="text-slate-600">Q1. 選ぶ主体が自分にあることが明示されていた</div>
              <input type="number" name="pc_i1" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q2. 無理のない範囲で調整してよい／自由に決めてよいと感じた</div>
              <input type="number" name="pc_i2" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
          </div>
        </div>

        <div class="p-3 bg-slate-50 rounded-xl border space-y-2 col-span-2">
          <div class="font-semibold text-slate-700">行動意図（1〜5, 2項目）</div>
          <div class="grid grid-cols-2 gap-2">
            <label class="block">
              <div class="text-slate-600">Q1. いま提示された提案を「やってみよう」と思う</div>
              <input type="number" name="int_i1" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
            <label class="block">
              <div class="text-slate-600">Q2. 近いうちに実行できそうだと感じる</div>
              <input type="number" name="int_i2" min="1" max="5" step="1" class="mt-1 border rounded px-2 py-1 text-sm w-full" required />
            </label>
          </div>
        </div>

        <label class="flex flex-col gap-1">
          <span>共感性・寄り添い感（1〜5）</span>
          <input type="number" name="perceived_empathy" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1">
          <span>理由・意味づけの明確さ（1〜5）</span>
          <input type="number" name="perceived_value_support" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1">
          <span>丁寧さ（1〜5）</span>
          <input type="number" name="perceived_politeness" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1 col-span-2">
          <span>自由記述（任意）</span>
          <textarea name="free_text" rows="3"
                    class="border rounded px-2 py-1 text-sm w-full"
                    placeholder="感じたこと・気になった点・数値では表しにくい印象があれば自由にお書きください"></textarea>
        </label>
      </div>

      <button type="submit" class="mt-1 px-3 py-2 rounded bg-sky-600 text-white text-sm">
        評価して終了
      </button>
    </form>
    """

    return f"""
    <div id="conversation" class="space-y-3">
      {header}
      <div class="p-3 bg-slate-50 rounded-xl border space-y-2 max-h-[420px] overflow-y-auto js-convo-box">
        {convo_html}
      </div>
      {advance_form}
    </div>
    """

@router.get("/ui", response_class=HTMLResponse)
def ui_home():
    """Main HTML page with session start + conversation UI."""
    order_method_value = 'LATIN'
    order_method_label = 'カウンタバランス'

    start_form = f"""
    <section class="p-4 bg-white rounded-xl shadow space-y-3">
      <h2 class="font-semibold">1. セッション開始</h2>
      <form
        hx-post="/ui/start_session"
        hx-target="#alerts"
        hx-swap="innerHTML"
        hx-indicator="#global-loading"
        class="space-y-2"
      >
        <label class="block text-sm text-slate-700">
          ゴール
          <input name="goal" class="mt-1 border rounded px-2 py-1 w-full text-sm"
                 placeholder="例: 英語学習を続けたい / 研究執筆を進めたい" />
        </label>

        <label class="block text-sm text-slate-700">
          コンテキスト（背景・前提・制約など）
          <textarea name="initial_context" rows="3"
                    class="mt-1 border rounded px-2 py-1 w-full text-sm"
                    placeholder="例: 平日は時間がなく、短時間でできる方法を探している など"></textarea>
        </label>

        <div class="flex flex-wrap gap-2 text-sm items-center">
          <span class="text-slate-700">命令性</span>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="directiveness" value="AUTO" checked class="accent-sky-600" />
            <span>AUTO（ランダム）</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="directiveness" value="HIGH" class="accent-sky-600" />
            <span>HIGH</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="directiveness" value="LOW" class="accent-sky-600" />
            <span>LOW</span>
          </label>
        </div>

        <div class="flex flex-wrap gap-2 text-sm items-center">
          <span class="text-slate-700">自由度提示</span>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="choice_framing" value="AUTO" checked class="accent-sky-600" />
            <span>AUTO（ランダム）</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="choice_framing" value="PRESENT" class="accent-sky-600" />
            <span>PRESENT</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="choice_framing" value="ABSENT" class="accent-sky-600" />
            <span>ABSENT</span>
          </label>
        </div>

        <div class="flex flex-wrap gap-2 text-sm items-center">
  <span class="text-slate-700">実験モード</span>
  <label class="inline-flex items-center gap-1">
    <input type="radio" name="design_mode" value="BETWEEN" class="accent-sky-600" />
    <span>被験者間（1条件固定）</span>
  </label>
  <label class="inline-flex items-center gap-1">
    <input type="radio" name="design_mode" value="WITHIN" checked class="accent-sky-600" />
    <span>被験者内（4条件）</span>
  </label>
  <span class="text-slate-700 ml-2">順序</span>
  <span class="text-slate-600 text-sm">ランダム</span>
</div>

<label class="inline-flex items-center gap-2 text-xs text-slate-600">
          <input type="checkbox" name="new_user" value="1" class="accent-sky-600" checked />
          <span>新しい被験者として開始（既存の user を使い回さない）</span>
        </label>
        <button type="submit" class="mt-1 px-3 py-1 rounded bg-emerald-600 text-white text-sm">
          セッションを開始（＋最初のメッセージ生成）
        </button>
      </form>
    </section>
    """

    convo = _render_conversation(None)

    body = start_form + f"""
    <section class="p-4 bg-white rounded-xl shadow space-y-3">
      <h2 class="font-semibold">2. 支援メッセージと評価</h2>
      {convo}
    </section>
    """

    return _wrap_page(body)


@router.post("/ui/start_session", response_class=HTMLResponse)
def ui_start_session(
    goal: str = Form(...),
    initial_context: str = Form(""),
    directiveness: str = Form("AUTO"),
    choice_framing: str = Form("AUTO"),
    design_mode: str = Form("BETWEEN"),
    new_user: Optional[str] = Form(None),
):
    """Start a session and generate the first instruction."""
    if not goal.strip():
        return '<div class="p-3 rounded bg-rose-100 text-rose-800 text-sm">ゴールを入力してください。</div>'

    def _resolve_d(v: str) -> Directiveness:
        v = (v or "AUTO").upper()
        if v == "AUTO":
            return random.choice([Directiveness.HIGH, Directiveness.LOW])
        return Directiveness(v)

    def _resolve_c(v: str) -> ChoiceFraming:
        v = (v or "AUTO").upper()
        if v == "AUTO":
            return random.choice([ChoiceFraming.PRESENT, ChoiceFraming.ABSENT])
        return ChoiceFraming(v)

    with SessionLocal() as s:
        if new_user:
            user = create_anonymous_user(s)
        else:
            user = get_or_create_singleton_user(s)

        d = _resolve_d(directiveness)
        c = _resolve_c(choice_framing)

        dm = (design_mode or "BETWEEN").upper()
        order_json = None
        cond_index = 0

        if dm == "WITHIN":
            conds = [
                {"directiveness": "HIGH", "choice_framing": "PRESENT"},
                {"directiveness": "HIGH", "choice_framing": "ABSENT"},
                {"directiveness": "LOW", "choice_framing": "PRESENT"},
                {"directiveness": "LOW", "choice_framing": "ABSENT"},
            ]
            # 順序はランダムに決定
            order = conds[:]
            random.shuffle(order)

            import json as _json
            order_json = _json.dumps(order, ensure_ascii=False)
            d = Directiveness(order[0]["directiveness"])
            c = ChoiceFraming(order[0]["choice_framing"])

        else:
            # BETWEEN (1 condition): keep the exact same condition-sequence machinery as WITHIN
            import json as _json
            order_json = _json.dumps([
                {"directiveness": d.value, "choice_framing": c.value}
            ], ensure_ascii=False)
            cond_index = 0

        es = create_session(
            s,
            user,
            goal=goal.strip(),
            directiveness=d,
            choice_framing=c,
            design_mode=dm,
            condition_order_json=order_json,
            condition_index=cond_index,
            initial_context=initial_context or None,
        )

        # first turn: no context
        # For within-subject design, fix content plan ONCE and reuse across conditions
        fixed_plan_json = getattr(es, 'fixed_task_plan_json', None)
        if not fixed_plan_json:
            _, plan = generate_content_plan(goal=es.goal, context_text=initial_context or '')
            fixed_plan_json = __import__('json').dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)
        d_curr, c_curr, _, _ = get_current_condition(es)
        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=d_curr,
            choice_framing=c_curr,
            context_text=initial_context or '',
            fixed_content_plan_json=fixed_plan_json,
        )
        log_turn(
            s,
            es,
            0,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
            task_plan_json=content_plan_json,
            num_options=integrity.get("num_options"),
            num_steps_total=integrity.get("num_steps_total"),
            char_count=integrity.get("char_count"),
            deviation_flags=flags,
            rerender_count=rerender_count,
        )

        convo_html = _render_conversation(es).replace('id="conversation"', 'id="conversation" hx-swap-oob="true"')
        alert = (
            f'<div class="p-3 rounded bg-emerald-100 text-emerald-800 text-sm">'
            f"セッションを開始しました（ID: {es.id}, 条件: {_label_directiveness(d)} / {_label_choice_framing(c)}）。"
            f"</div>"
        )
        return alert + convo_html


@router.post("/ui/advance_turn", response_class=HTMLResponse)
def ui_advance_turn(
    session_id: int = Form(...),
    turn_id: int = Form(...),
    # evaluation inputs (required)
    autonomy_i1: int = Form(...),
    autonomy_i2: int = Form(...),
    autonomy_i3: int = Form(...),

    coercion_i1: int = Form(...),
    coercion_i2: int = Form(...),
    coercion_i3: int = Form(...),
    coercion_i4: int = Form(...),

    pd_i1: int = Form(...),
    pd_i2: int = Form(...),

    pc_i1: int = Form(...),
    pc_i2: int = Form(...),

    int_i1: int = Form(...),
    int_i2: int = Form(...),

    perceived_empathy: Optional[float] = Form(None),
    perceived_value_support: Optional[float] = Form(None),
    perceived_politeness: Optional[float] = Form(None),
    free_text: Optional[str] = Form(None),
):
    """Save evaluation for the latest turn, advance condition (within-subject), and generate the next instruction."""

    with SessionLocal() as s:
        # 1) Save evaluation for the turn just shown
        try:
            update_evaluation(
                s,
                turn_id,
                autonomy_items=[autonomy_i1, autonomy_i2, autonomy_i3],
                coercion_items=[coercion_i1, coercion_i2, coercion_i3, coercion_i4],
                perceived_directiveness_items=[pd_i1, pd_i2],
                perceived_choice_items=[pc_i1, pc_i2],
                intention_items=[int_i1, int_i2],
                perceived_empathy=perceived_empathy,
                perceived_value_support=perceived_value_support,
                perceived_politeness=perceived_politeness,
                free_text=free_text,
            )
        except ValueError:
            return '<div class="p-3 rounded bg-amber-100 text-amber-800 text-sm">評価対象のターンが見つかりませんでした。</div>'

        # 2) Advance condition for within-subject design
        advance_condition(s, session_id)

        # 3) Generate next instruction (same as ui_next_turn)
        es = get_session(s, session_id)

        # If within-subject conditions are complete, do not generate any further messages.
        if within_is_complete(es):
            return _render_conversation(es)

        context_lines = []
        last_log = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        if last_log is not None:
            context_lines.append(f"直前の提案(LLM): {last_log.llm_output}")
            if last_log.user_response:
                context_lines.append(f"直前のユーザー返答: {last_log.user_response}")
        context_text = "\n".join(context_lines)

        last = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        next_idx = 0 if last is None else last.turn_index + 1

        d_curr, c_curr, cond_idx, cond_total = get_current_condition(es)
        fixed_plan_json = getattr(es, 'fixed_task_plan_json', None)
        if not fixed_plan_json:
            _, plan = generate_content_plan(goal=es.goal, context_text=initial_context or '')
            fixed_plan_json = __import__('json').dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)

        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=d_curr,
            choice_framing=c_curr,
            context_text=context_text,
            fixed_content_plan_json=fixed_plan_json,
        )

        log_turn(
            s,
            es,
            next_idx,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
            task_plan_json=content_plan_json,
            num_options=integrity.get("num_options"),
            num_steps_total=integrity.get("num_steps_total"),
            char_count=integrity.get("char_count"),
            deviation_flags=flags,
            rerender_count=rerender_count,
            directiveness_override=d_curr.value,
            choice_framing_override=c_curr.value,
            user_response="",
            action_choice="",
        )

        return _render_conversation(es)

@router.post("/ui/next_turn", response_class=HTMLResponse)
def ui_next_turn(
    session_id: int = Form(...),
):
    """Generate next instruction and refresh conversation block."""

    with SessionLocal() as s:
        es = get_session(s, session_id)

        # If within-subject conditions are complete, do not generate any further messages.
        if within_is_complete(es):
            return _render_conversation(es)

        context_lines = []

        # Include previous turn (instruction + user's reply stored on that turn)
        last_log = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        if last_log is not None:
            context_lines.append(f"直前の提案(LLM): {last_log.llm_output}")
            if last_log.user_response:
                context_lines.append(f"直前のユーザー返答: {last_log.user_response}")

        # Also include the current user's reply and action choice
        context_text = "\n".join(context_lines)

        last = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        next_idx = 0 if last is None else last.turn_index + 1

        d_curr, c_curr, cond_idx, cond_total = get_current_condition(es)

        fixed_plan_json = getattr(es, 'fixed_task_plan_json', None)
        if not fixed_plan_json:
            _, plan = generate_content_plan(goal=es.goal, context_text='')
            fixed_plan_json = __import__('json').dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)

        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=d_curr,
            choice_framing=c_curr,
            context_text=context_text,
            fixed_content_plan_json=fixed_plan_json,
        )

        log_turn(
            s,
            es,
            next_idx,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
            task_plan_json=content_plan_json,
            num_options=integrity.get("num_options"),
            num_steps_total=integrity.get("num_steps_total"),
            char_count=integrity.get("char_count"),
            deviation_flags=flags,
            rerender_count=rerender_count,
            directiveness_override=d_curr.value,
            choice_framing_override=c_curr.value,
            user_response="",
            action_choice="",
        )

        # Re-render conversation
        return _render_conversation(es)


@router.post("/ui/evaluate_turn", response_class=HTMLResponse)
def ui_evaluate_turn(
    session_id: int = Form(...),
    turn_id: Optional[int] = Form(None),

    autonomy_i1: int = Form(...),
    autonomy_i2: int = Form(...),
    autonomy_i3: int = Form(...),

    coercion_i1: int = Form(...),
    coercion_i2: int = Form(...),
    coercion_i3: int = Form(...),
    coercion_i4: int = Form(...),

    pd_i1: int = Form(...),
    pd_i2: int = Form(...),

    pc_i1: int = Form(...),
    pc_i2: int = Form(...),

    int_i1: int = Form(...),
    int_i2: int = Form(...),

    perceived_empathy: Optional[float] = Form(None),
    perceived_value_support: Optional[float] = Form(None),
    perceived_politeness: Optional[float] = Form(None),
    free_text: Optional[str] = Form(None),
):
    """Save evaluation scores for a given turn (typically the last one)."""
    if not turn_id:
        return '<div class="p-3 rounded bg-amber-100 text-amber-800 text-sm">評価対象のターンが見つかりませんでした。</div>'

    with SessionLocal() as s:
        try:
            update_evaluation(
                s,
                turn_id,
                autonomy_items=[autonomy_i1, autonomy_i2, autonomy_i3],
                coercion_items=[coercion_i1, coercion_i2, coercion_i3, coercion_i4],
                perceived_directiveness_items=[pd_i1, pd_i2],
                perceived_choice_items=[pc_i1, pc_i2],
                intention_items=[int_i1, int_i2],
                perceived_empathy=perceived_empathy,
                perceived_value_support=perceived_value_support,
                perceived_politeness=perceived_politeness,
                free_text=free_text,
            )
        except ValueError as e:
            return f'<div class="p-3 rounded bg-rose-100 text-rose-800 text-sm">{html.escape(str(e))}</div>'

    # Note: Condition advancement is handled by /ui/advance_turn to keep evaluation and generation coupled.
    return '<div class="p-3 rounded bg-emerald-100 text-emerald-800 text-sm">評価を保存しました。</div>'
