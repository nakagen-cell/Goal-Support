from __future__ import annotations

import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Literal

from .schemas import (
    StartSessionIn,
    StartSessionOut,
    NextInstructionIn,
    NextInstructionOut,
    EvaluationIn,
    EvaluationOut,
)
from .repo import (
    SessionLocal,
    init_db,
    create_anonymous_user,
    get_or_create_singleton_user,
    create_session,
    get_session,
    log_turn,
    update_evaluation,
    export_logs_as_csv,
)
from .models import Style, TurnLog
from .llm import generate_instruction
from . import webui as _webui_router

app = FastAPI(
    title="LLM Instruction Style Control Prototype",
    description=(
        "LLMを用いた目標達成支援におけるインストラクション文体制御が"
        "自律性認知・「やらされ感」および行動意図に与える影響を検討するための"
        "対話型評価プラットフォーム用バックエンドAPI"
    ),
)

# mount web UI router
app.include_router(_webui_router.router)


@app.on_event("startup")
def _on_startup() -> None:
    init_db()


@app.get("/llm/status")
def llm_status():
    """Return a simple health-check style status for the LLM backend."""
    present = bool(os.getenv("OPENAI_API_KEY"))
    return {"ok": present, "provider": "openai", "env_key_present": present}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _resolve_style(style_in: Literal["DIRECTIVE", "SUGGESTIVE", "COLLABORATIVE", "AUTO"]) -> Style:
    if style_in == "AUTO":
        return random.choice([Style.DIRECTIVE, Style.SUGGESTIVE, Style.COLLABORATIVE])
    return Style(style_in)


@app.post("/session/start", response_model=StartSessionOut)
def start_session(req: StartSessionIn):
    """Start an experimental session.

    - Decide (or randomly assign) an instruction style
    - Register a session row
    - Generate the first instruction
    """
    with SessionLocal() as s:
        if req.new_user:
            user = create_anonymous_user(s)
        else:
            # Single shared user for quick local testing; swap as needed.
            user = get_or_create_singleton_user(s)

        style = _resolve_style(req.style)
        es = create_session(s, user, goal=req.goal, style=style)

        llm_prompt, llm_output, task_plan_json = generate_instruction(
            goal=es.goal,
            style=es.condition_style,
            context_text="",
        )
        log_turn(
            s,
            es,
            0,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
        task_plan_json=task_plan_json,
        )

        return StartSessionOut(
            session_id=es.id,
            user_id=user.id,
            style=style.value,  # type: ignore[arg-type]
            goal=es.goal,
        )


# ---------------------------------------------------------------------------
# Turn-level interaction
# ---------------------------------------------------------------------------

@app.post("/session/next", response_model=NextInstructionOut)
def next_instruction(req: NextInstructionIn):
    """Generate the next instruction for a given session and log the turn."""
    with SessionLocal() as s:
        es = get_session(s, req.session_id)

        # Context text is deliberately simple; the frontend can decide when
        # to include which pieces of information.
        context_lines = []
        if req.user_response:
            context_lines.append(f"ユーザーからの直近の返答: {req.user_response}")
        if req.action_choice:
            context_lines.append(f"ユーザーの行動選択: {req.action_choice}")
        context_text = "\n".join(context_lines)

        # Determine turn index (0-based)
        last = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        next_idx = 0 if last is None else last.turn_index + 1

        llm_prompt, llm_output, task_plan_json = generate_instruction(
            goal=es.goal,
            style=es.condition_style,
            context_text=context_text,
        )

        log = log_turn(
            s,
            es,
            next_idx,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
        task_plan_json=task_plan_json,
            user_response=req.user_response or "",
            action_choice=req.action_choice or "",
        )

        return NextInstructionOut(
            session_id=es.id,
            turn_id=log.id,
            turn_index=log.turn_index,
            style=es.condition_style.value,  # type: ignore[arg-type]
            instruction=log.llm_output,
        )


# ---------------------------------------------------------------------------
# Evaluation logging
# ---------------------------------------------------------------------------

@app.post("/session/evaluate", response_model=EvaluationOut)
def evaluate_turn(req: EvaluationIn):
    """Attach psychological evaluation scores to a logged turn.

    自律性・やらされ感・文体知覚指標・行動意図の評価値を，指定されたターンに付与する。
    """
    with SessionLocal() as s:
        try:
            log = update_evaluation(
                s,
                req.turn_id,
                autonomy_score=req.autonomy_score,
                coercion_score=req.coercion_score,
                perceived_directiveness=req.perceived_directiveness,
                perceived_choice=req.perceived_choice,
                perceived_empathy=req.perceived_empathy,
                perceived_value_support=req.perceived_value_support,
                intention_score=req.intention_score,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        return EvaluationOut(
            turn_id=log.id,
            autonomy_score=log.autonomy_score,
            coercion_score=log.coercion_score,
            perceived_directiveness=log.perceived_directiveness,
            perceived_choice=log.perceived_choice,
            perceived_empathy=log.perceived_empathy,
            perceived_value_support=log.perceived_value_support,
            intention_score=log.intention_score,
        )


# ---------------------------------------------------------------------------
# Log export
# ---------------------------------------------------------------------------

@app.get("/export/logs.csv", response_class=PlainTextResponse)
def export_logs():
    """Export all logs as CSV for analysis (条件比較・媒介分析などに利用)。"""
    with SessionLocal() as s:
        csv_text = export_logs_as_csv(s)
    return PlainTextResponse(content=csv_text or "", media_type="text/csv")
