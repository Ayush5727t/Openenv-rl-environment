from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    abuse = "abuse"


class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class SupportTeam(str, Enum):
    billing_ops = "billing_ops"
    tech_support = "tech_support"
    trust_safety = "trust_safety"
    account_support = "account_support"


class ActionType(str, Enum):
    classify = "classify"
    set_priority = "set_priority"
    assign_team = "assign_team"
    reorder_queue = "reorder_queue"
    escalate = "escalate"
    draft_response = "draft_response"
    resolve = "resolve"
    noop = "noop"


class TicketGroundTruth(BaseModel):
    ticket_id: str
    subject: str
    customer_message: str
    correct_category: TicketCategory
    correct_priority: TicketPriority
    correct_team: SupportTeam
    requires_escalation: bool = False
    required_response_keywords: List[str] = Field(default_factory=list)


class TicketAgentState(BaseModel):
    ticket_id: str
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[TicketPriority] = None
    predicted_team: Optional[SupportTeam] = None
    escalated: bool = False
    drafted_response: str = ""
    resolved: bool = False


class TicketObservation(BaseModel):
    ticket_id: str
    subject: str
    customer_message: str
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[TicketPriority] = None
    predicted_team: Optional[SupportTeam] = None
    escalated: bool = False
    resolved: bool = False


class TaskSpec(BaseModel):
    task_id: Literal["easy", "medium", "hard"]
    title: str
    max_steps: int = Field(gt=0)
    tickets: List[TicketGroundTruth] = Field(min_length=1)
    reward_weights: Dict[str, float] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    task_id: Literal["easy", "medium", "hard"]
    step_count: int = 0
    max_steps: int
    queue_order: List[str]
    ground_truth: Dict[str, TicketGroundTruth]
    tickets: Dict[str, TicketAgentState]
    total_reward: float = 0.0
    done: bool = False
    action_history: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: Literal["easy", "medium", "hard"]
    step_count: int
    remaining_steps: int
    queue_order: List[str]
    tickets: List[TicketObservation]
    available_actions: List[ActionType]
    instruction: str


class Action(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    value: Optional[str] = None
    target_index: Optional[int] = None
    response_text: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    shaped_delta: float
    task_score: float = Field(ge=0.0, le=1.0)
    penalties: Dict[str, float] = Field(default_factory=dict)


class StepInfo(BaseModel):
    task_score: float = Field(ge=0.0, le=1.0)
    penalties: Dict[str, float] = Field(default_factory=dict)
    invalid_action: bool = False
    message: str = ""
