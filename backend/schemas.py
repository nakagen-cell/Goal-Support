from __future__ import annotations
from pydantic import BaseModel
from typing import Literal, Optional


StyleLiteral = Literal["DIRECTIVE", "SUGGESTIVE", "COLLABORATIVE"]


class StartSessionIn(BaseModel):
    goal: str
    # AUTO = backend randomly assigns one of the three styles.
    style: Literal["DIRECTIVE", "SUGGESTIVE", "COLLABORATIVE", "AUTO"] = "AUTO"
    # If true, a new anonymous user will always be created.
    new_user: bool = False


class StartSessionOut(BaseModel):
    session_id: int
    user_id: int
    style: StyleLiteral
    goal: str


class NextInstructionIn(BaseModel):
    session_id: int
    # Optional free-form text from the user (may be empty for the first turn).
    user_response: Optional[str] = None
    # Optional categorical choice like "DO" / "POSTPONE" / "ADJUST" / "REST".
    action_choice: Optional[str] = None


class NextInstructionOut(BaseModel):
    session_id: int
    turn_id: int
    turn_index: int
    style: StyleLiteral
    instruction: str


class EvaluationIn(BaseModel):
    """心理評価および文体知覚評価の入力スキーマ。"""

    session_id: int
    turn_id: int
    autonomy_score: Optional[float] = None
    coercion_score: Optional[float] = None
    perceived_directiveness: Optional[float] = None
    perceived_choice: Optional[float] = None
    perceived_empathy: Optional[float] = None
    perceived_value_support: Optional[float] = None
    intention_score: Optional[float] = None


class EvaluationOut(BaseModel):
    """心理評価および文体知覚評価の出力スキーマ。"""

    turn_id: int
    autonomy_score: Optional[float]
    coercion_score: Optional[float]
    perceived_directiveness: Optional[float]
    perceived_choice: Optional[float]
    perceived_empathy: Optional[float]
    perceived_value_support: Optional[float]
    intention_score: Optional[float]
