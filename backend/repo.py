from __future__ import annotations
from typing import Optional, Iterable, Sequence
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, User, ExperimentSession, TurnLog, Directiveness, ChoiceFraming
import csv
import io
import json

# SQLite DB. For experiments you may wish to change the path.
engine = create_engine("sqlite:///./goal_support.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not exist yet, and apply lightweight migrations."""
    Base.metadata.create_all(bind=engine)

    # Lightweight migration for SQLite: add new columns if missing.
    # (We keep existing *_score columns as the canonical mean fields.)
    with engine.begin() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info('turn_logs')").fetchall()
        existing = {c[1] for c in cols}  # (cid, name, type, ...)

        def ensure(col_name: str, col_type_sql: str) -> None:
            if col_name in existing:
                return
            conn.exec_driver_sql(f"ALTER TABLE turn_logs ADD COLUMN {col_name} {col_type_sql}")

        ensure("autonomy_items_json", "TEXT")
        ensure("coercion_items_json", "TEXT")
        ensure("perceived_directiveness_items_json", "TEXT")
        ensure("perceived_choice_items_json", "TEXT")
        ensure("intention_items_json", "TEXT")
        ensure("free_text", "TEXT")
        ensure("perceived_politeness", "REAL")

    # Migrations for sessions table (within-subject mode support)
    with engine.begin() as conn2:
        cols2 = conn2.exec_driver_sql("PRAGMA table_info('sessions')").fetchall()
        existing2 = {c[1] for c in cols2}

        def ensure_s(col_name: str, col_type_sql: str) -> None:
            if col_name in existing2:
                return
            conn2.exec_driver_sql(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type_sql}")

        ensure_s("design_mode", "TEXT")
        ensure_s("condition_order_json", "TEXT")
        ensure_s("condition_index", "INTEGER")
        ensure_s("fixed_task_plan_json", "TEXT")
        ensure_s("initial_context", "TEXT")


# ---------------------------------------------------------------------------
# User & session helpers
# ---------------------------------------------------------------------------

def create_anonymous_user(session) -> User:
    """Create a new anonymous user row.

    実験参加者を匿名 ID のみで管理するための簡易ヘルパー。
    個人情報は保持しない。
    """
    user = User()
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_or_create_singleton_user(session) -> User:
    """Return a single shared 'user' row (for quick local testing).

    既に複数の User 行が存在しても、最も ID が小さい 1 件を拾う。
    """
    u = session.query(User).order_by(User.id.asc()).first()
    if u:
        return u
    return create_anonymous_user(session)


def create_session(
    session,
    user: User,
    *,
    goal: str,
    directiveness: Directiveness,
    choice_framing: ChoiceFraming,
    design_mode: str = "BETWEEN",
    condition_order_json: str | None = None,
    condition_index: int = 0,
    initial_context: str | None = None,
    fixed_task_plan_json: str | None = None,
) -> ExperimentSession:
    """Create a new experimental session row."""
    es = ExperimentSession(
        user_id=user.id,
        goal=goal,
        directiveness=directiveness,
        choice_framing=choice_framing,
        design_mode=design_mode,
        condition_order_json=condition_order_json,
        condition_index=condition_index,
        initial_context=initial_context,
        fixed_task_plan_json=fixed_task_plan_json,
    )
    session.add(es)
    session.commit()
    session.refresh(es)
    return es


def get_session(session, session_id: int) -> ExperimentSession:
    es = session.get(ExperimentSession, session_id)
    if not es:
        raise ValueError(f"session {session_id} not found")
    return es


# ---------------------------------------------------------------------------
# Turn logging
# ---------------------------------------------------------------------------

def log_turn(
    session,
    es: ExperimentSession,
    turn_index: int,
    *,
    llm_prompt: str,
    llm_output: str,
    directiveness_override: str | None = None,
    choice_framing_override: str | None = None,
    task_plan_json: str | None = None,
    # integrity
    num_options: int | None = None,
    num_steps_total: int | None = None,
    char_count: int | None = None,
    # deviation
    deviation_flags: Sequence[str] | None = None,
    rerender_count: int = 0,
    # user inputs
    user_response: str = "",
    action_choice: str = "",
) -> TurnLog:
    """Insert one log row for a single LLM–user interaction turn."""
    log = TurnLog(
        session_id=es.id,
        turn_index=turn_index,
        directiveness=(directiveness_override or es.directiveness.value),
        choice_framing=(choice_framing_override or es.choice_framing.value),
        llm_prompt=llm_prompt,
        llm_output=llm_output,
        task_plan_json=task_plan_json,
        num_options=num_options,
        num_steps_total=num_steps_total,
        char_count=char_count,
        deviation_flags=json.dumps(list(deviation_flags or []), ensure_ascii=False),
        rerender_count=int(rerender_count or 0),
        user_response=user_response or "",
        action_choice=action_choice or "",
    )
    session.add(log)
    session.commit()
    session.refresh(log)
    return log


def update_evaluation(
    session,
    turn_id: int,
    *,
    autonomy_items: Sequence[float],
    coercion_items: Sequence[float],
    perceived_directiveness_items: Sequence[float],
    perceived_choice_items: Sequence[float],
    intention_items: Sequence[float],
    perceived_empathy: float | None = None,
    perceived_value_support: float | None = None,
    perceived_politeness: float | None = None,
    free_text: str | None = None,
) -> TurnLog:
    """Update evaluation for a given turn.

    - Save item-level responses as JSON arrays
    - Compute means and store them in *_score / perceived_* columns
    """
    log = session.get(TurnLog, turn_id)
    if not log:
        raise ValueError(f"turn {turn_id} not found")

    def _mean(xs: Sequence[float]) -> float:
        xs2 = [float(x) for x in xs]
        if not xs2:
            raise ValueError("empty item list")
        return sum(xs2) / len(xs2)

    # Raw item responses (JSON)
    log.autonomy_items_json = json.dumps(list(autonomy_items), ensure_ascii=False)
    log.coercion_items_json = json.dumps(list(coercion_items), ensure_ascii=False)
    log.perceived_directiveness_items_json = json.dumps(list(perceived_directiveness_items), ensure_ascii=False)
    log.perceived_choice_items_json = json.dumps(list(perceived_choice_items), ensure_ascii=False)
    log.intention_items_json = json.dumps(list(intention_items), ensure_ascii=False)

    # Means
    log.autonomy_score = _mean(autonomy_items)
    log.coercion_score = _mean(coercion_items)
    log.perceived_directiveness = _mean(perceived_directiveness_items)
    log.perceived_choice = _mean(perceived_choice_items)
    log.intention_score = _mean(intention_items)

    if perceived_empathy is not None:
        log.perceived_empathy = float(perceived_empathy)
    if perceived_value_support is not None:
        log.perceived_value_support = float(perceived_value_support)

    if perceived_politeness is not None:
        log.perceived_politeness = float(perceived_politeness)

    if free_text is not None:
        log.free_text = str(free_text)

    session.add(log)
    session.commit()
    session.refresh(log)
    return log


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def iter_joined_logs(session) -> Iterable[dict]:
    """Yield log rows joined with user and session info as dictionaries."""
    q = (
        session.query(TurnLog, ExperimentSession, User)
        .join(ExperimentSession, TurnLog.session_id == ExperimentSession.id)
        .join(User, ExperimentSession.user_id == User.id)
        .order_by(TurnLog.timestamp.asc(), TurnLog.id.asc())
    )
    for log, es, user in q.all():
        yield {
            "user_id": user.id,
            "session_id": es.id,
            # IMPORTANT: For within-subject designs, the condition can vary per turn.
            # We therefore export BOTH the per-turn condition (canonical for analysis)
            # and the session-level defaults (useful for debugging).
            "directiveness": log.directiveness,
            "choice_framing": log.choice_framing,
            "session_directiveness": es.directiveness.value,
            "session_choice_framing": es.choice_framing.value,
            "goal": es.goal,
            "initial_context": getattr(es, "initial_context", None),
            "design_mode": getattr(es, "design_mode", "BETWEEN"),
            "condition_order_json": getattr(es, "condition_order_json", None),
            "condition_index": getattr(es, "condition_index", 0),
            "turn_id": log.id,
            "turn_index": log.turn_index,
            "llm_prompt": log.llm_prompt,
            "llm_output": log.llm_output,
            "content_plan_json": log.task_plan_json,
            "num_options": log.num_options,
            "num_steps_total": log.num_steps_total,
            "char_count": log.char_count,
            "deviation_flags": log.deviation_flags,
            "rerender_count": log.rerender_count,
            "user_response": log.user_response,
            "action_choice": log.action_choice,
            "autonomy_score": log.autonomy_score,
            "coercion_score": log.coercion_score,
            "perceived_directiveness": log.perceived_directiveness,
            "perceived_choice": log.perceived_choice,
            "perceived_empathy": log.perceived_empathy,
            "perceived_value_support": log.perceived_value_support,
            "perceived_politeness": log.perceived_politeness,
            "intention_score": log.intention_score,
            "free_text": log.free_text,
            "timestamp": log.timestamp.isoformat(),
        }


def export_logs_as_csv(session) -> str:
    """Return all joined logs as a CSV string."""
    rows = list(iter_joined_logs(session))
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Within-subject helpers
# ---------------------------------------------------------------------------

def get_current_condition(es: ExperimentSession) -> tuple[Directiveness, ChoiceFraming, int, int]:
    """Return (directiveness, choice_framing, idx, total) for the current condition.

    We primarily rely on condition_order_json if present. This allows BETWEEN sessions
    (single condition) to share the same condition sequencing logic as WITHIN sessions.
    """
    order_json = getattr(es, 'condition_order_json', None)
    if order_json:
        try:
            order = __import__('json').loads(order_json) or []
            if isinstance(order, list) and len(order) > 0:
                total = len(order)
                idx = int(getattr(es, 'condition_index', 0) or 0)
                if idx < 0:
                    idx = 0
                # When idx == total, we consider the session "completed"; show the last condition.
                show_idx = idx if idx < total else (total - 1)
                cond = order[show_idx] or {}
                d = Directiveness(str(cond.get('directiveness', es.directiveness)).upper())
                c = ChoiceFraming(str(cond.get('choice_framing', es.choice_framing)).upper())
                return d, c, min(idx, total), total
        except Exception:
            pass

    # Fallback: no order_json
    return es.directiveness, es.choice_framing, 0, 1

def get_within_total(es: ExperimentSession) -> int:
    """Return the number of conditions for the session.

    Historically this returned the number of *within-subject* conditions and 0 otherwise.
    To make BETWEEN (1-condition) behave exactly like WITHIN in UI + processing, we treat
    any session that has a condition_order_json as having a condition sequence.

    If condition_order_json is missing or invalid, we fall back to:
      - 4 for WITHIN (legacy behavior)
      - 1 otherwise
    """
    order_json = getattr(es, 'condition_order_json', None)
    if order_json:
        try:
            order = __import__('json').loads(order_json) or []
            if isinstance(order, list) and len(order) > 0:
                return len(order)
        except Exception:
            pass

    # Legacy fallback
    if (getattr(es, 'design_mode', 'BETWEEN') or 'BETWEEN').upper() == 'WITHIN':
        try:
            legacy_order = __import__('json').loads(getattr(es, 'condition_order_json', None) or '[]') or []
            if isinstance(legacy_order, list) and len(legacy_order) > 0:
                return len(legacy_order)
        except Exception:
            pass
        return 4

    return 1

def within_is_complete(es: ExperimentSession) -> bool:
    """True if the session is within-subject and has advanced past the last condition.

    We treat condition_index == total as "completed".
    """
    total = get_within_total(es)
    if total <= 0:
        return False
    idx = int(getattr(es, 'condition_index', 0) or 0)
    return idx >= total

def advance_condition(session, session_id: int) -> ExperimentSession:
    es = session.get(ExperimentSession, session_id)
    if not es:
        raise ValueError(f"session {session_id} not found")
    # Advance whenever a condition sequence exists (condition_order_json). This makes BETWEEN (1 condition)
    # behave exactly like WITHIN in progression/termination.
    if not getattr(es, 'condition_order_json', None):
        return es

    total = get_within_total(es)
    idx = int(getattr(es, 'condition_index', 0) or 0)
    # Advance by 1, but do not grow unbounded.
    # We allow idx == total to represent "completed" (one past the last valid condition).
    next_idx = idx + 1
    if total > 0 and next_idx > total:
        next_idx = total

    es.condition_index = next_idx
    session.add(es)
    session.commit()
    session.refresh(es)
    return es


def set_fixed_task_plan(session, session_id: int, fixed_task_plan_json: str) -> ExperimentSession:
    """Persist a session-level fixed content plan (used for within-subject designs).

    We store the plan once and then reuse it across conditions so that only
    expression parameters vary.
    """
    es = session.get(ExperimentSession, session_id)
    if not es:
        raise ValueError(f"session {session_id} not found")
    es.fixed_task_plan_json = fixed_task_plan_json
    session.add(es)
    session.commit()
    session.refresh(es)
    return es
