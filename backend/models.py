from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, ForeignKey, DateTime, Text, Enum, Float
from datetime import datetime
import enum


class Base(DeclarativeBase):
    """SQLAlchemy base class."""
    pass


class Style(enum.Enum):
    """Instruction style conditions used in the experiments.

    本研究で用いる 3 種類のインストラクション文体条件：

    - DIRECTIVE: 指示スタイル（命令的で断定的な文体）
    - SUGGESTIVE: 提案スタイル（選択肢提示と自律性支持を行う文体）
    - COLLABORATIVE: 協働スタイル（質問と共感的対話を含む文体）
    """

    DIRECTIVE = "DIRECTIVE"
    SUGGESTIVE = "SUGGESTIVE"
    COLLABORATIVE = "COLLABORATIVE"


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
    """One experimental session for a given user and style condition.

    1 つのユーザーと 1 つの文体条件（指示／提案／協働）の組み合わせで構成される
    対話エピソードを表す。論文中の「対話型プロトタイプ実験」や短期フィールド試験の
    1 セッションに対応する。"""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    condition_style: Mapped[Style] = mapped_column(Enum(Style))
    goal: Mapped[str] = mapped_column(Text)  # user-defined goal text
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class TurnLog(Base):
    """Log for each LLM–user interaction turn.

    論文第 5 章の「ログ設計」で挙げた項目に対応する。各ターンについて，次を記録する。

    - user_id（User 経由）
    - condition_style（ExperimentSession 経由）
    - goal（ExperimentSession 経由）
    - llm_prompt / llm_output
    - user_response
    - action_choice（実行／延期／調整／休養）
    - autonomy_score（自律性）
    - coercion_score（やらされ感）
    - perceived_directiveness（命令性）
    - perceived_choice（選択自由度）
    - perceived_empathy（共感性）
    - perceived_value_support（価値・理由提示の明確性）
    - intention_score（行動意図）
    - timestamp
    """

    __tablename__ = "turn_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    turn_index: Mapped[int] = mapped_column(Integer)  # 0, 1, 2, ...

    llm_prompt: Mapped[str] = mapped_column(Text)
    llm_output: Mapped[str] = mapped_column(Text)

    # Style-agnostic task plan stored as JSON
    task_plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Raw text sent from the user for this turn (if any)
    user_response: Mapped[str] = mapped_column(Text, default="")
    # Categorical action choice (e.g., DO / POSTPONE / ADJUST / REST)
    action_choice: Mapped[str] = mapped_column(String(32), default="")

    # Main psychological evaluation indices (optional, filled after the turn)
    autonomy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coercion_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 文体の背後要因を測定する補助指標
    perceived_directiveness: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_choice: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_empathy: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_value_support: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 行動意図（例：「この支援を今後も使いたい」「今日の提案に従って行動したい」などの平均）
    intention_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session = relationship("ExperimentSession")
    user = relationship("User", secondary="sessions", viewonly=True)
