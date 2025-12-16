from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
from typing import Optional
from .repo import (
    SessionLocal,
    create_anonymous_user,
    get_or_create_singleton_user,
    create_session,
    get_session,
    log_turn,
    update_evaluation,
)
from .models import Style, TurnLog, ExperimentSession
from .llm import generate_instruction
import html

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
  <title>Instruction Style Prototype UI</title>
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
    <h1 class="text-2xl font-bold">LLM インストラクション文体制御 UI</h1>
    <p class="text-sm text-slate-600">
      指示 / 提案 / 協働スタイルを切り替えて、対話と主観評価を収集するための簡易 Web UI です。
      （サーバー処理中は画面上部に <span class="font-mono">LOAD...</span> が表示されます）
    </p>
    <div id="alerts"></div>
    {body}
  </div>
</body>
</html>"""


def _style_label(style: Style) -> str:
    if style == Style.DIRECTIVE:
        return "指示スタイル (DIRECTIVE)"
    if style == Style.SUGGESTIVE:
        return "提案スタイル (SUGGESTIVE)"
    if style == Style.COLLABORATIVE:
        return "協働スタイル (COLLABORATIVE)"
    return style.value


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

    header = f"""
    <div class="p-3 bg-slate-100 rounded-lg text-xs text-slate-700">
      <div>セッションID: <span class="font-mono">{session_id}</span></div>
      <div>スタイル: <span class="font-semibold">{html.escape(_style_label(es.condition_style))}</span></div>
      <div>ゴール: {html.escape(es.goal)}</div>
    </div>
    """

    bubbles = []
    for log in logs:
        # System/LLM message
        bubbles.append(
            f'<div class="flex justify-start">'
            f'<div class="max-w-[80%] p-3 my-1 rounded-xl bg-white shadow text-sm whitespace-pre-line">'
            f'<div class="text-xs text-slate-400 mb-1">LLM</div>'
            f'{html.escape(log.llm_output)}</div></div>'
        )
        if log.user_response:
            bubbles.append(
                f'<div class="flex justify-end">'
                f'<div class="max-w-[80%] p-3 my-1 rounded-xl bg-sky-600 text-white text-sm">'
                f'<div class="text-xs text-sky-100 mb-1">You</div>'
                f'{html.escape(log.user_response)}</div></div>'
            )

    convo_html = "".join(bubbles) or """
      <div class="text-sm text-slate-500">
        まだメッセージはありません。「次のメッセージを生成」から開始してください。
      </div>
    """

    # last turn for evaluation form
    last_turn_id = logs[-1].id if logs else ""

    eval_form = f"""
    <form
      hx-post="/ui/evaluate_turn"
      hx-target="#alerts"
      hx-swap="innerHTML"
      hx-indicator="#global-loading"
      class="mt-3 space-y-2"
    >
      <input type="hidden" name="session_id" value="{session_id}" />
      <input type="hidden" name="turn_id" value="{last_turn_id}" />
      <div class="text-sm font-semibold text-slate-700">このセッションの直近メッセージに対する評価</div>
      <div class="grid grid-cols-2 gap-2 text-xs items-center">
        <label class="flex flex-col gap-1">
          <span>自律性（1〜5）</span>
          <input type="number" name="autonomy_score" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1">
          <span>やらされ感（1〜5）</span>
          <input type="number" name="coercion_score" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1">
          <span>命令性の強さ（1〜5）</span>
          <input type="number" name="perceived_directiveness" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
        <label class="flex flex-col gap-1">
          <span>選択の自由度（1〜5）</span>
          <input type="number" name="perceived_choice" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
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
        <label class="flex flex-col gap-1 col-span-2">
          <span>行動意図（1〜5）</span>
          <input type="number" name="intention_score" min="1" max="5" step="1"
                 class="border rounded px-2 py-1 text-sm" />
        </label>
      </div>
      <button type="submit" class="mt-1 px-3 py-1 rounded bg-emerald-600 text-white text-sm">
        評価を保存
      </button>
    </form>
    """

    input_form = f"""
    <form
      hx-post="/ui/next_turn"
      hx-target="#conversation"
      hx-swap="outerHTML"
      hx-indicator="#global-loading"
      class="mt-3 space-y-2"
    >
      <input type="hidden" name="session_id" value="{session_id}" />
      <label class="block text-sm text-slate-700">
        あなたの返答（任意）
        <textarea name="user_response" rows="2" class="mt-1 border rounded px-2 py-1 w-full text-sm"
                  placeholder="LLMへの返答や状態メモなど"></textarea>
      </label>
      <label class="block text-sm text-slate-700">
        行動選択（任意）
        <select name="action_choice" class="mt-1 border rounded px-2 py-1 text-sm">
          <option value="">（未選択）</option>
          <option value="DO">実行する</option>
          <option value="POSTPONE">延期する</option>
          <option value="ADJUST">内容を調整する</option>
          <option value="REST">今日は休む</option>
        </select>
      </label>
      <button type="submit" class="px-3 py-1 rounded bg-sky-600 text-white text-sm">
        次のメッセージを生成
      </button>
    </form>
    """

    return f"""
    <div id="conversation" class="space-y-3">
      {header}
      <div class="p-3 bg-slate-50 rounded-xl border space-y-2 max-h-[420px] overflow-y-auto">
        {convo_html}
      </div>
      {input_form}
      {eval_form}
    </div>
    """


@router.get("/ui", response_class=HTMLResponse)
def ui_home():
    """Main HTML page with session start + conversation UI."""
    start_form = """
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
        <div class="flex flex-wrap gap-2 text-sm items-center">
          <span class="text-slate-700">スタイル</span>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="style" value="AUTO" checked class="accent-sky-600" />
            <span>AUTO（ランダム割当）</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="style" value="DIRECTIVE" class="accent-sky-600" />
            <span>指示</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="style" value="SUGGESTIVE" class="accent-sky-600" />
            <span>提案</span>
          </label>
          <label class="inline-flex items-center gap-1">
            <input type="radio" name="style" value="COLLABORATIVE" class="accent-sky-600" />
            <span>協働</span>
          </label>
        </div>
        <label class="inline-flex items-center gap-2 text-xs text-slate-600">
          <input type="checkbox" name="new_user" value="1" class="accent-sky-600" />
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
      <h2 class="font-semibold">2. 対話と評価</h2>
      {convo}
    </section>
    """

    return _wrap_page(body)


@router.post("/ui/start_session", response_class=HTMLResponse)
def ui_start_session(
    goal: str = Form(...),
    style: str = Form("AUTO"),
    new_user: Optional[str] = Form(None),
):
    """Start a session and generate the first instruction."""
    style = style or "AUTO"
    if style not in ("AUTO", "DIRECTIVE", "SUGGESTIVE", "COLLABORATIVE"):
        style = "AUTO"

    if not goal.strip():
        return '<div class="p-3 rounded bg-rose-100 text-rose-800 text-sm">ゴールを入力してください。</div>'

    with SessionLocal() as s:
        if new_user:
            user = create_anonymous_user(s)
        else:
            user = get_or_create_singleton_user(s)

        if style == "AUTO":
            import random as _r
            st = _r.choice([Style.DIRECTIVE, Style.SUGGESTIVE, Style.COLLABORATIVE])
        else:
            st = Style(style)

        es = create_session(s, user, goal=goal.strip(), style=st)

        # first turn: no context
        llm_prompt, llm_output = generate_instruction(goal=es.goal, style=st, context_text="")
        log_turn(s, es, 0, llm_prompt=llm_prompt, llm_output=llm_output)

        convo_html = _render_conversation(es).replace('id="conversation"', 'id="conversation" hx-swap-oob="true"')
        alert = (
            f'<div class="p-3 rounded bg-emerald-100 text-emerald-800 text-sm">'
            f"セッションを開始しました（ID: {es.id}, スタイル: {_style_label(st)}）。"
            f"</div>"
        )
        return alert + convo_html


@router.post("/ui/next_turn", response_class=HTMLResponse)
def ui_next_turn(
    session_id: int = Form(...),
    user_response: Optional[str] = Form(None),
    action_choice: Optional[str] = Form(None),
):
    """Generate next instruction and refresh conversation block."""
    with SessionLocal() as s:
        es = get_session(s, session_id)

        context_lines = []
        if user_response:
            context_lines.append(f"ユーザーからの直近の返答: {user_response}")
        if action_choice:
            context_lines.append(f"ユーザーの行動選択: {action_choice}")
        context_text = "\n".join(context_lines)

        last = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        next_idx = 0 if last is None else last.turn_index + 1

        llm_prompt, llm_output = generate_instruction(
            goal=es.goal,
            style=es.condition_style,
            context_text=context_text,
        )

        log_turn(
            s,
            es,
            next_idx,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
            user_response=user_response or "",
            action_choice=action_choice or "",
        )

        # Re-render conversation
        return _render_conversation(es)


@router.post("/ui/evaluate_turn", response_class=HTMLResponse)
def ui_evaluate_turn(
    session_id: int = Form(...),
    turn_id: Optional[int] = Form(None),
    autonomy_score: Optional[float] = Form(None),
    coercion_score: Optional[float] = Form(None),
    perceived_directiveness: Optional[float] = Form(None),
    perceived_choice: Optional[float] = Form(None),
    perceived_empathy: Optional[float] = Form(None),
    perceived_value_support: Optional[float] = Form(None),
    intention_score: Optional[float] = Form(None),
):
    """Save evaluation scores for a given turn (typically the last one)."""
    if not turn_id:
        return '<div class="p-3 rounded bg-amber-100 text-amber-800 text-sm">評価対象のターンが見つかりませんでした。</div>'

    with SessionLocal() as s:
        try:
            update_evaluation(
                s,
                turn_id,
                autonomy_score=autonomy_score,
                coercion_score=coercion_score,
                perceived_directiveness=perceived_directiveness,
                perceived_choice=perceived_choice,
                perceived_empathy=perceived_empathy,
                perceived_value_support=perceived_value_support,
                intention_score=intention_score,
            )
        except ValueError as e:
            return f'<div class="p-3 rounded bg-rose-100 text-rose-800 text-sm">{html.escape(str(e))}</div>'

    return '<div class="p-3 rounded bg-emerald-100 text-emerald-800 text-sm">評価を保存しました。</div>'
