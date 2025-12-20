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
    AdvanceTurnIn,
    AdvanceTurnOut,
)
from .repo import (
    get_current_condition,
    within_is_complete,

    SessionLocal,
    init_db,
    create_anonymous_user,
    get_or_create_singleton_user,
    create_session,
    get_session,
    log_turn,
    update_evaluation,
    export_logs_as_csv,
    set_fixed_task_plan,
)
from .models import Directiveness, ChoiceFraming, TurnLog
from .llm import generate_instruction, generate_content_plan
from . import webui as _webui_router

app = FastAPI(
    title="LLM Instruction Expression Control Prototype",
    description=(
        "LLM生成支援メッセージにおける『やらされ感（統制知覚）』低減のため、"
        "命令性×自由度提示（2×2）を因果操作して評価する対話型実験プラットフォーム用バックエンドAPI"
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

def _resolve_directiveness(d_in: Literal["HIGH", "LOW", "AUTO"]) -> Directiveness:
    if d_in == "AUTO":
        return random.choice([Directiveness.HIGH, Directiveness.LOW])
    return Directiveness(d_in)


def _resolve_choice_framing(c_in: Literal["PRESENT", "ABSENT", "AUTO"]) -> ChoiceFraming:
    if c_in == "AUTO":
        return random.choice([ChoiceFraming.PRESENT, ChoiceFraming.ABSENT])
    return ChoiceFraming(c_in)


@app.post("/session/start", response_model=StartSessionOut)
def start_session(req: StartSessionIn):
    """Start an experimental session.

    - Decide (or randomly assign) (directiveness, choice_framing)
    - Register a session row
    - Generate the first instruction (content plan -> verbalizer)
    """
    with SessionLocal() as s:
        if req.new_user:
            user = create_anonymous_user(s)
        else:
            # Single shared user for quick local testing; swap as needed.
            user = get_or_create_singleton_user(s)

        # Decide conditions
        d = _resolve_directiveness(req.directiveness)
        c = _resolve_choice_framing(req.choice_framing)

        design_mode = "BETWEEN"
        order_json = None
        cond_index = 0

        if getattr(req, "within_subject", False):
            design_mode = "WITHIN"
            conds = [
                {"directiveness": "HIGH", "choice_framing": "PRESENT"},
                {"directiveness": "HIGH", "choice_framing": "ABSENT"},
                {"directiveness": "LOW", "choice_framing": "PRESENT"},
                {"directiveness": "LOW", "choice_framing": "ABSENT"},
            ]            # 順序はランダム（セッション開始ごとにシャッフル）
            order = conds[:]
            random.shuffle(order)
            order_json = __import__("json").dumps(order, ensure_ascii=False)
            d = Directiveness(order[0]["directiveness"])
            c = ChoiceFraming(order[0]["choice_framing"])
        else:
            # BETWEEN (1 condition): use the same condition sequencing machinery as WITHIN
            import json as _json
            order_json = _json.dumps([{"directiveness": d.value, "choice_framing": c.value}], ensure_ascii=False)
            cond_index = 0


        es = create_session(
            s,
            user,
            goal=req.goal,
            directiveness=d,
            choice_framing=c,
            design_mode=design_mode,
            condition_order_json=order_json,
            condition_index=cond_index,
            initial_context=req.initial_context,
        )

        # For within-subject design, fix the content plan ONCE and reuse it across conditions.
        fixed_plan_json: str | None = None
        if design_mode == "WITHIN":
            _, plan = generate_content_plan(goal=es.goal, context_text=getattr(req, 'initial_context', '') or "")
            fixed_plan_json = __import__("json").dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)

        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=es.directiveness,
            choice_framing=es.choice_framing,
            context_text=getattr(req, 'initial_context', '') or "",
            fixed_content_plan_json=fixed_plan_json,
        )

        log_turn(
            s,
            es,
            0,
            llm_prompt=llm_prompt,
            llm_output=llm_output,
            directiveness_override=es.directiveness.value,
            choice_framing_override=es.choice_framing.value,
            task_plan_json=content_plan_json,
            num_options=integrity.get("num_options"),
            num_steps_total=integrity.get("num_steps_total"),
            char_count=integrity.get("char_count"),
            deviation_flags=flags,
            rerender_count=rerender_count,
        )

        return StartSessionOut(
            session_id=es.id,
            user_id=user.id,
            directiveness=es.directiveness.value,  # type: ignore[arg-type]
            choice_framing=es.choice_framing.value,  # type: ignore[arg-type]
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

        # Guard: do not allow generating further turns after within-subject completion
        if within_is_complete(es):
            raise HTTPException(status_code=409, detail="within-subject conditions are complete")

        # Context text: include at least the immediately previous instruction + the user's reply
        # so the planner can stay grounded in the ongoing exchange.
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

        # Also include the current user's reply (provided by the caller) and action choice
        context_text = "\n".join(context_lines)

        # Determine turn index (0-based)
        last = (
            s.query(TurnLog)
            .filter_by(session_id=es.id)
            .order_by(TurnLog.turn_index.desc())
            .first()
        )
        next_idx = 0 if last is None else last.turn_index + 1

        d_curr, c_curr, cond_idx, cond_total = get_current_condition(es)

        fixed_plan_json = getattr(es, "fixed_task_plan_json", None)
        if not fixed_plan_json:
            # Safety fallback: create and persist a fixed plan if missing.
            _, plan = generate_content_plan(goal=es.goal, context_text="")
            fixed_plan_json = __import__("json").dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)

        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=d_curr,
            choice_framing=c_curr,
            context_text=context_text,
            fixed_content_plan_json=fixed_plan_json,
        )

        log = log_turn(
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

        return NextInstructionOut(
            session_id=es.id,
            turn_id=log.id,
            turn_index=log.turn_index,
            directiveness=d_curr.value,  # type: ignore[arg-type]
            choice_framing=c_curr.value,  # type: ignore[arg-type]
            instruction=log.llm_output,
        )


# ---------------------------------------------------------------------------
# Evaluation logging
# ---------------------------------------------------------------------------

@app.post("/session/evaluate", response_model=EvaluationOut)
def evaluate_turn(req: EvaluationIn):
    """Attach psychological evaluation scores to a logged turn."""
    with SessionLocal() as s:
        try:
            log = update_evaluation(
                s,
                req.turn_id,
                autonomy_items=req.autonomy_items,
                coercion_items=req.coercion_items,
                perceived_directiveness_items=req.perceived_directiveness_items,
                perceived_choice_items=req.perceived_choice_items,
                intention_items=req.intention_items,
                perceived_empathy=req.perceived_empathy,
                perceived_value_support=req.perceived_value_support,
                perceived_politeness=req.perceived_politeness,
                free_text=req.free_text,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Parse JSON item arrays (stored as text) for output
        import json as _json

        def _loads(sv):
            try:
                return _json.loads(sv) if sv else []
            except Exception:
                return []

        return EvaluationOut(
            turn_id=log.id,
            autonomy_items=_loads(getattr(log, "autonomy_items_json", None)),
            coercion_items=_loads(getattr(log, "coercion_items_json", None)),
            perceived_directiveness_items=_loads(getattr(log, "perceived_directiveness_items_json", None)),
            perceived_choice_items=_loads(getattr(log, "perceived_choice_items_json", None)),
            intention_items=_loads(getattr(log, "intention_items_json", None)),
            autonomy_score=float(log.autonomy_score or 0.0),
            coercion_score=float(log.coercion_score or 0.0),
            perceived_directiveness=float(log.perceived_directiveness or 0.0),
            perceived_choice=float(log.perceived_choice or 0.0),
            intention_score=float(log.intention_score or 0.0),
            perceived_empathy=log.perceived_empathy,
            perceived_value_support=log.perceived_value_support,
            perceived_politeness=getattr(log, "perceived_politeness", None),
            free_text=getattr(log, "free_text", None),
        )



# ---------------------------------------------------------------------------
# Combined: evaluate + next
# ---------------------------------------------------------------------------

@app.post("/session/advance", response_model=AdvanceTurnOut)
def advance_turn(req: AdvanceTurnIn):
    """Save evaluation for a given turn, advance condition (within-subject), then generate the next instruction."""
    with SessionLocal() as s:
        # 1) Save evaluation
        try:
            update_evaluation(
                s,
                req.turn_id,
                autonomy_items=req.autonomy_items,
                coercion_items=req.coercion_items,
                perceived_directiveness_items=req.perceived_directiveness_items,
                perceived_choice_items=req.perceived_choice_items,
                intention_items=req.intention_items,
                perceived_empathy=req.perceived_empathy,
                perceived_value_support=req.perceived_value_support,
                perceived_politeness=req.perceived_politeness,
                free_text=req.free_text,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # 2) Advance condition for within-subject design (so the *next* message uses the next condition)
        from .repo import advance_condition
        advance_condition(s, req.session_id)

        # 3) Generate next instruction (same logic as /session/next, plus logging)
        es = get_session(s, req.session_id)

        # Guard: do not allow generating further turns after within-subject completion
        if within_is_complete(es):
            raise HTTPException(status_code=409, detail="within-subject conditions are complete")

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

        d_curr, c_curr, _, _ = get_current_condition(es)

        fixed_plan_json = getattr(es, "fixed_task_plan_json", None)
        if not fixed_plan_json:
            _, plan = generate_content_plan(goal=es.goal, context_text="")
            fixed_plan_json = __import__("json").dumps(plan, ensure_ascii=False)
            set_fixed_task_plan(s, es.id, fixed_plan_json)

        llm_prompt, llm_output, content_plan_json, integrity, flags, rerender_count = generate_instruction(
            goal=es.goal,
            directiveness=d_curr,
            choice_framing=c_curr,
            context_text=context_text,
            fixed_content_plan_json=fixed_plan_json,
        )

        log = log_turn(
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

        return AdvanceTurnOut(
            session_id=es.id,
            evaluated_turn_id=req.turn_id,
            next_turn_id=log.id,
            next_turn_index=log.turn_index,
            directiveness=d_curr.value,  # type: ignore[arg-type]
            choice_framing=c_curr.value,  # type: ignore[arg-type]
            instruction=log.llm_output,
        )

# ---------------------------------------------------------------------------
# Log export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Log export (JSON)
# ---------------------------------------------------------------------------

@app.get("/export/logs.json")
def export_logs_json():
    """Export all logs as JSON (list of dicts) for analysis."""
    import json as _json
    with SessionLocal() as s:
        rows = list(iter_joined_logs(s))
    return rows

@app.get("/export/logs.csv", response_class=PlainTextResponse)
def export_logs():
    """Export all logs as CSV for analysis (条件比較・媒介分析などに利用)。"""
    with SessionLocal() as s:
        csv_text = export_logs_as_csv(s)
    return PlainTextResponse(content=csv_text or "", media_type="text/csv")
