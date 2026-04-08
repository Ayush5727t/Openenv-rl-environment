"""
OpenEnv Models - Pydantic models for observations, actions, rewards, and state.

This module defines the core data structures used throughout the OpenEnv environment.
All models use Pydantic for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Enumeration of possible action types in the environment."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_LIST = "file_list"
    API_CALL = "api_call"
    DATA_PROCESS = "data_process"
    TEXT_EDIT = "text_edit"
    COMMAND_EXEC = "command_exec"
    TASK_COMPLETE = "task_complete"


class Action(BaseModel):
    """
    Represents an action taken by an agent in the environment.
    
    Attributes:
        action_type: The type of action being performed
        parameters: Dictionary of parameters specific to the action type
        timestamp: When the action was created
    """
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ObservationType(str, Enum):
    """Types of observations the environment can provide."""
    TEXT = "text"
    FILE_CONTENT = "file_content"
    FILE_LIST = "file_list"
    API_RESPONSE = "api_response"
    ERROR = "error"
    SUCCESS = "success"
    STATE_UPDATE = "state_update"


class Observation(BaseModel):
    """
    Represents an observation from the environment after an action.
    
    Attributes:
        observation_type: Type of observation
        content: The actual observation data (text, file content, etc.)
        metadata: Additional contextual information
        timestamp: When the observation was created
    """
    observation_type: ObservationType
    content: Union[str, Dict[str, Any], List[Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RewardType(str, Enum):
    """Types of rewards that can be given."""
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETION = "task_completion"
    EFFICIENCY = "efficiency"
    CORRECTNESS = "correctness"
    PENALTY = "penalty"


class Reward(BaseModel):
    """
    Represents a reward signal from the environment.
    
    Attributes:
        value: Numeric reward value (can be negative for penalties)
        reward_type: Category of the reward
        reason: Human-readable explanation of why this reward was given
        metadata: Additional reward-related data
    """
    value: float
    reward_type: RewardType
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskStatus(str, Enum):
    """Status of a task in the environment."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """
    Information about the current task.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type/category of the task
        description: Human-readable task description
        status: Current task status
        progress: Progress percentage (0-100)
        steps_taken: Number of steps taken so far
        max_steps: Maximum allowed steps (None for unlimited)
        metadata: Additional task-specific data
    """
    task_id: str
    task_type: str
    description: str
    status: TaskStatus = TaskStatus.NOT_STARTED
    progress: float = Field(default=0.0)
    steps_taken: int = 0
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('progress')
    def validate_progress(cls, v):
        """Ensure progress is between 0 and 100."""
        return max(0.0, min(100.0, v))


class EnvironmentState(BaseModel):
    """
    Complete state of the environment at a given point in time.
    
    Attributes:
        task_info: Information about the current task
        episode_id: Unique identifier for the current episode
        step_count: Total steps in current episode
        total_reward: Cumulative reward in current episode
        history: List of recent actions and observations
        workspace: Current workspace state (files, variables, etc.)
        is_terminal: Whether the episode has ended
        timestamp: When this state was captured
    """
    task_info: TaskInfo
    episode_id: str
    step_count: int = 0
    total_reward: float = 0.0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    workspace: Dict[str, Any] = Field(default_factory=dict)
    is_terminal: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_to_history(self, action: Action, observation: Observation, reward: Reward):
        """Add a step to the history."""
        self.history.append({
            "step": self.step_count,
            "action": action.model_dump(),
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 100 steps in memory
        if len(self.history) > 100:
            self.history = self.history[-100:]


class StepResult(BaseModel):
    """
    Result of a single step in the environment.
    
    This is returned by the step() method and contains all information
    about what happened during the step.
    
    Attributes:
        observation: What the agent observes after the action
        reward: Reward signal for the action
        done: Whether the episode has ended
        info: Additional information about the step
    """
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """
    Result of resetting the environment.
    
    Attributes:
        observation: Initial observation after reset
        info: Information about the new episode/task
    """
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)
