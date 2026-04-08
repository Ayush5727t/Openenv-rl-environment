"""
Task System - Abstract base classes and registry for managing tasks.

This module provides the infrastructure for creating and managing different
types of tasks in the OpenEnv environment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from models.models import (
    Action, Observation, Reward, TaskInfo, TaskStatus,
    ObservationType, RewardType
)
import uuid


class Task(ABC):
    """
    Abstract base class for all tasks in the environment.
    
    Each task represents a specific challenge or goal that an agent
    must accomplish. Tasks define:
    - Initial state/setup
    - How actions affect the task
    - How to calculate rewards
    - When the task is complete
    """
    
    def __init__(self, task_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new task.
        
        Args:
            task_id: Optional unique identifier (auto-generated if not provided)
            config: Task-specific configuration
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.config = config or {}
        self.status = TaskStatus.NOT_STARTED
        self.steps_taken = 0
        self.task_data: Dict[str, Any] = {}
        self._initialize_task()
    
    @abstractmethod
    def _initialize_task(self):
        """Initialize task-specific state and data. Called during __init__."""
        pass
    
    @abstractmethod
    def get_task_info(self) -> TaskInfo:
        """
        Get current information about this task.
        
        Returns:
            TaskInfo object with current task state
        """
        pass
    
    @abstractmethod
    def get_initial_observation(self) -> Observation:
        """
        Get the initial observation when the task starts.
        
        Returns:
            Initial observation describing the task
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: Action) -> Observation:
        """
        Execute an action and return the resulting observation.
        
        Args:
            action: The action to execute
            
        Returns:
            Observation resulting from the action
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, action: Action, observation: Observation) -> Reward:
        """
        Calculate the reward for an action and its observation.
        
        Args:
            action: The action that was taken
            observation: The observation resulting from the action
            
        Returns:
            Reward object with value and explanation
        """
        pass
    
    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if the task is complete.
        
        Returns:
            True if task is successfully completed
        """
        pass
    
    @abstractmethod
    def is_failed(self) -> bool:
        """
        Check if the task has failed.
        
        Returns:
            True if task has failed and cannot be completed
        """
        pass
    
    def reset(self):
        """Reset the task to its initial state."""
        self.status = TaskStatus.NOT_STARTED
        self.steps_taken = 0
        self.task_data.clear()
        self._initialize_task()
    
    def update_status(self):
        """Update the task status based on current state."""
        if self.is_complete():
            self.status = TaskStatus.COMPLETED
        elif self.is_failed():
            self.status = TaskStatus.FAILED
        elif self.steps_taken > 0:
            self.status = TaskStatus.IN_PROGRESS
    
    @property
    def max_steps(self) -> Optional[int]:
        """Maximum number of steps allowed for this task."""
        return self.config.get('max_steps')
    
    @property
    def task_type(self) -> str:
        """Return the type/name of this task."""
        return self.__class__.__name__


class TaskRegistry:
    """
    Registry for managing available task types.
    
    This allows dynamic task creation and discovery of available tasks.
    """
    
    _tasks: Dict[str, Type[Task]] = {}
    
    @classmethod
    def register(cls, task_name: str):
        """
        Decorator to register a task class.
        
        Usage:
            @TaskRegistry.register("file_task")
            class FileTask(Task):
                ...
        """
        def decorator(task_class: Type[Task]):
            cls._tasks[task_name] = task_class
            return task_class
        return decorator
    
    @classmethod
    def create_task(cls, task_name: str, task_id: Optional[str] = None, 
                   config: Optional[Dict[str, Any]] = None) -> Task:
        """
        Create a task instance by name.
        
        Args:
            task_name: Name of the registered task type
            task_id: Optional task ID
            config: Task configuration
            
        Returns:
            New task instance
            
        Raises:
            KeyError: If task_name is not registered
        """
        if task_name not in cls._tasks:
            raise KeyError(f"Task '{task_name}' not found. Available: {list(cls._tasks.keys())}")
        
        task_class = cls._tasks[task_name]
        return task_class(task_id=task_id, config=config)
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """Get list of all registered task names."""
        return list(cls._tasks.keys())
    
    @classmethod
    def get_task_class(cls, task_name: str) -> Type[Task]:
        """Get the task class for a given task name."""
        if task_name not in cls._tasks:
            raise KeyError(f"Task '{task_name}' not found")
        return cls._tasks[task_name]


class TaskValidator:
    """
    Utility class for validating task actions and constraints.
    """
    
    @staticmethod
    def validate_action(task: Task, action: Action) -> tuple[bool, str]:
        """
        Validate if an action is allowed for the current task state.
        
        Args:
            task: The task to validate against
            action: The action to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if task is in a valid state for actions
        if task.status == TaskStatus.COMPLETED:
            return False, "Task is already completed"
        
        if task.status == TaskStatus.FAILED:
            return False, "Task has failed"
        
        # Check max steps
        if task.max_steps and task.steps_taken >= task.max_steps:
            return False, f"Maximum steps ({task.max_steps}) exceeded"
        
        return True, ""
    
    @staticmethod
    def validate_task_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate task configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if 'max_steps' in config:
            max_steps = config['max_steps']
            if not isinstance(max_steps, int) or max_steps < 1:
                return False, "max_steps must be a positive integer"
        
        return True, ""
