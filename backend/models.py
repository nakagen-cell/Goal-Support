from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, ForeignKey, DateTime, Text, Enum, Float
from datetime import datetime
import enum


class Base(DeclarativeBase):
    """SQLAlchemy base class."""
    pass


class Directiveness(enum.Enum):
    """Expression parameter: directiveness (imperative-ness)."""
    HIGH = "HIGH"
    LOW = "LOW"


class ChoiceFraming(enum.Enum):
    """Expression parameter: choice-framing / freedom-of-choice statement."""
    PRESENT = "PRESENT"  # freedom-of-choice wording is explicitly present
    ABSENT = "ABSENT"    # no explicit wording about user choice


class User(Base):
    """Anonymous user / participant.

    実験参加者を表すエンティティ。個人情報は保持せず，匿名 ID のみを管理する。
    多くの実験設定ではブラウザセッション 1 つが 1 ユーザーに対応する。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Optional label for debugging; not used for analysis
    name: Mapped[str] = mapped_column(String(64), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExperimentSession(Base):
    """One experimental session for a given user and experimental condition (d, c).

    本研究の 2×2 因子計画：
    - Directiveness (HIGH / LOW)
    - ChoiceFraming (PRESENT / ABSENT)
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    directiveness: Mapped[Directiveness] = mapped_column(Enum(Directiveness))
    choice_framing: Mapped[ChoiceFraming] = mapped_column(Enum(ChoiceFraming))

    # Experimental design mode
    design_mode: Mapped[str] = mapped_column(String(16), default="BETWEEN")  # BETWEEN or WITHIN
    # For within-subject: JSON list of condition dicts [{directiveness, choice_framing}, ...]
    condition_order_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # For within-subject: which condition index is currently active (0..3)
    condition_index: Mapped[int] = mapped_column(Integer, default=0)

    goal: Mapped[str] = mapped_column(Text)  # user-defined goal text

    initial_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # For within-subject: fixed content plan (JSON) shared across conditions
    fixed_task_plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class TurnLog(Base):
    """Log for each LLM–user interaction turn."""

    __tablename__ = "turn_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    turn_index: Mapped[int] = mapped_column(Integer)  # 0, 1, 2, ...

    # Condition snapshot for analysis convenience
    directiveness: Mapped[str] = mapped_column(String(16), default="")
    choice_framing: Mapped[str] = mapped_column(String(16), default="")

    llm_prompt: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)

    # Content plan stored as JSON (content planner output)
    task_plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content integrity metrics
    num_options: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_steps_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Deviation management
    deviation_flags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    rerender_count: Mapped[int] = mapped_column(Integer, default=0)

    # Raw text sent from the user for this turn (if any)
    user_response: Mapped[str] = mapped_column(Text, default="")
    # Categorical action choice (e.g., DO / POSTPONE / ADJUST / REST)
    action_choice: Mapped[str] = mapped_column(String(32), default="")

    # Main psychological evaluation indices (optional, filled after the turn)
    autonomy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coercion_score: Mapped[float | None] = mapped_column(Float, nullable=True)


    # Multi-item scale responses stored as JSON arrays (e.g., [1,2,3]).
    autonomy_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    coercion_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    perceived_directiveness_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    perceived_choice_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    intention_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Manipulation checks / perceived covariates
    perceived_directiveness: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_choice: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_empathy: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_value_support: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_politeness: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional free text comment for the turn
    free_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Behavioral intention (optional)
    intention_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session = relationship("ExperimentSession")
    user = relationship("User", secondary="sessions", viewonly=True)
