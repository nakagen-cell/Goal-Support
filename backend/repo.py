from __future__ import annotations
from typing import Optional, Iterable
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from .models import Base, User, ExperimentSession, TurnLog, Style
import csv
import io

# SQLite DB. For experiments you may wish to change the path.
engine = create_engine("sqlite:///./goal_support.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


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

    もともとの実装では scalar_one_or_none() を使っていましたが、
    既に複数の User 行が存在すると MultipleResultsFound エラーになるため、
    「最も ID が小さい 1 件を拾う」実装に変更しています。
    """
    # 複数ユーザーがいても、最初の 1 行だけを使う
    u = session.query(User).order_by(User.id.asc()).first()
    if u:
        return u
    return create_anonymous_user(session)


def create_session(session, user: User, goal: str, style: Style) -> ExperimentSession:
    """Create a new experimental session row."""
    es = ExperimentSession(user_id=user.id, goal=goal, condition_style=style)
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
    task_plan_json: str | None = None,   # ←★追加
    user_response: str = "",
    action_choice: str = "",
) -> TurnLog:
    """Insert one log row for a single LLM–user interaction turn."""
    log = TurnLog(
        session_id=es.id,
        turn_index=turn_index,
        llm_prompt=llm_prompt,
        llm_output=llm_output,
        task_plan_json=task_plan_json,   # ←★追加
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
    autonomy_score: float | None = None,
    coercion_score: float | None = None,
    perceived_directiveness: float | None = None,
    perceived_choice: float | None = None,
    perceived_empathy: float | None = None,
    perceived_value_support: float | None = None,
    intention_score: float | None = None,
) -> TurnLog:
    """Update psychological evaluation scores for a given turn.

    論文第 6 章で述べた評価指標：
    - 自律性
    - やらされ感
    - 文体知覚指標（命令性・選択自由度・共感性・価値理由提示）
    - 行動意図

    を 1 行のログと結び付けて保存する。
    """
    log = session.get(TurnLog, turn_id)
    if not log:
        raise ValueError(f"turn {turn_id} not found")

    if autonomy_score is not None:
        log.autonomy_score = float(autonomy_score)
    if coercion_score is not None:
        log.coercion_score = float(coercion_score)
    if perceived_directiveness is not None:
        log.perceived_directiveness = float(perceived_directiveness)
    if perceived_choice is not None:
        log.perceived_choice = float(perceived_choice)
    if perceived_empathy is not None:
        log.perceived_empathy = float(perceived_empathy)
    if perceived_value_support is not None:
        log.perceived_value_support = float(perceived_value_support)
    if intention_score is not None:
        log.intention_score = float(intention_score)

    session.add(log)
    session.commit()
    session.refresh(log)
    return log


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def iter_joined_logs(session) -> Iterable[dict]:
    """Yield log rows joined with user and session info as dictionaries.

    Section 5.4（ログ設計）で示した CSV レイアウトに対応する構造を返す。
    """
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
            "condition_style": es.condition_style.value,
            "goal": es.goal,
            "turn_id": log.id,
            "turn_index": log.turn_index,
            "llm_prompt": log.llm_prompt,
            "llm_output": log.llm_output,
            "user_response": log.user_response,
            "action_choice": log.action_choice,
            "autonomy_score": log.autonomy_score,
            "coercion_score": log.coercion_score,
            "perceived_directiveness": log.perceived_directiveness,
            "perceived_choice": log.perceived_choice,
            "perceived_empathy": log.perceived_empathy,
            "perceived_value_support": log.perceived_value_support,
            "intention_score": log.intention_score,
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
