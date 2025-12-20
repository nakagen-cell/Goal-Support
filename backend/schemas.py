from __future__ import annotations

from pydantic import BaseModel, conlist
from typing import Literal, Optional


DirectivenessLiteral = Literal["HIGH", "LOW"]
ChoiceFramingLiteral = Literal["PRESENT", "ABSENT"]

ActionChoiceLiteral = Literal["EXECUTE", "NOT_EXECUTE"]


class StartSessionIn(BaseModel):
    goal: str
    # Optional session-level context (saved; sent to LLM only at session start)
    initial_context: Optional[str] = None
    # If true, run within-subject 4-condition sequence (order randomized/counterbalanced).
    within_subject: bool = False
    # AUTO = backend randomly assigns one of the 2×2 conditions.
    directiveness: Literal["HIGH", "LOW", "AUTO"] = "AUTO"
    choice_framing: Literal["PRESENT", "ABSENT", "AUTO"] = "AUTO"
    # If true, a new anonymous user will always be created.
    new_user: bool = False


class StartSessionOut(BaseModel):
    session_id: int
    user_id: int
    directiveness: DirectivenessLiteral
    choice_framing: ChoiceFramingLiteral
    goal: str


class NextInstructionIn(BaseModel):
    session_id: int
    # Optional free-form text from the user (may be empty).
    user_response: Optional[str] = None
    # Required categorical choice: always choose one for each turn after the message is shown.
    action_choice: ActionChoiceLiteral


class NextInstructionOut(BaseModel):
    session_id: int
    turn_id: int
    turn_index: int
    directiveness: DirectivenessLiteral
    choice_framing: ChoiceFramingLiteral
    instruction: str


class EvaluationIn(BaseModel):
    """心理評価および操作チェックの入力スキーマ（互換用の単一スコア入力は廃止）。"""

    session_id: int
    turn_id: int

    # Multi-item scales (1〜5)
    # NOTE: Pydantic v2 uses min_length/max_length (not min_items/max_items).
    autonomy_items: conlist(float, min_length=3, max_length=3)
    coercion_items: conlist(float, min_length=4, max_length=4)  # やらされ感（統制知覚）
    perceived_directiveness_items: conlist(float, min_length=2, max_length=2)
    perceived_choice_items: conlist(float, min_length=2, max_length=2)
    intention_items: conlist(float, min_length=2, max_length=2)

    # Perceived covariates (single-item, optional)
    perceived_empathy: Optional[float] = None
    perceived_value_support: Optional[float] = None
    perceived_politeness: Optional[float] = None

    # Optional free-text comment
    free_text: Optional[str] = None


class EvaluationOut(BaseModel):
    """心理評価および操作チェックの出力スキーマ（平均値＋生データ）。"""

    turn_id: int

    autonomy_items: list[float]
    coercion_items: list[float]
    perceived_directiveness_items: list[float]
    perceived_choice_items: list[float]
    intention_items: list[float]

    # Means (stored also in *_score columns)
    autonomy_score: float
    coercion_score: float
    perceived_directiveness: float
    perceived_choice: float
    intention_score: float

    perceived_empathy: Optional[float]
    perceived_value_support: Optional[float]
    perceived_politeness: Optional[float]
    free_text: Optional[str]


# ---------------------------------------------------------------------------
# Combined action: evaluate the last turn and generate the next instruction
# ---------------------------------------------------------------------------

class AdvanceTurnIn(BaseModel):
    """Evaluate a turn and advance to generate the next instruction in one request."""

    session_id: int
    # The turn being evaluated (typically the most recent turn shown to the user)
    turn_id: int

    # Multi-item scales (1〜5)
    autonomy_items: conlist(float, min_length=3, max_length=3)
    coercion_items: conlist(float, min_length=4, max_length=4)
    perceived_directiveness_items: conlist(float, min_length=2, max_length=2)
    perceived_choice_items: conlist(float, min_length=2, max_length=2)
    intention_items: conlist(float, min_length=2, max_length=2)

    perceived_empathy: Optional[float] = None
    perceived_value_support: Optional[float] = None
    perceived_politeness: Optional[float] = None
    free_text: Optional[str] = None

    # Next-turn inputs
    user_response: Optional[str] = None
    action_choice: ActionChoiceLiteral


class AdvanceTurnOut(BaseModel):
    """Return the newly generated next instruction (and echo which turn was evaluated)."""

    session_id: int
    evaluated_turn_id: int
    next_turn_id: int
    next_turn_index: int
    directiveness: DirectivenessLiteral
    choice_framing: ChoiceFramingLiteral
    instruction: str
