"""
OpenEnv Environment - Main environment class implementing the OpenEnv API.

This module provides the core Environment class that implements the standard
step() / reset() / state() API for agent interaction.
"""

from typing import Any, Dict, Optional
from pathlib import Path
from models.models import (
    Action, Observation, Reward, EnvironmentState,
    StepResult, ResetResult, ObservationType
)
from tasks.base_task import Task, TaskRegistry, TaskValidator
from src.state_manager import StateManager


class OpenEnvEnvironment:
    """
    Main OpenEnv environment implementing the standard agent API.
    
    This environment provides a standardized interface for agents to:
    - Reset to a new task
    - Take actions via step()
    - Query current state
    
    The environment manages tasks, state, and reward calculation.
    """
    
    def __init__(
        self,
        default_task: Optional[str] = None,
        task_config: Optional[Dict[str, Any]] = None,
        persistence_dir: Optional[str] = None,
        max_steps_per_episode: int = 1000
    ):
        """
        Initialize the OpenEnv environment.
        
        Args:
            default_task: Name of default task type to use
            task_config: Configuration for task creation
            persistence_dir: Directory for state persistence
            max_steps_per_episode: Maximum steps per episode
        """
        self.default_task = default_task
        self.task_config = task_config or {}
        self.max_steps_per_episode = max_steps_per_episode
        
        # Initialize state manager
        persist_path = Path(persistence_dir) if persistence_dir else None
        self.state_manager = StateManager(persistence_dir=persist_path)
        
        # Current task
        self.current_task: Optional[Task] = None
        self._episode_active = False
    
    def reset(
        self,
        task_name: Optional[str] = None,
        task_config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> ResetResult:
        """
        Reset the environment to start a new episode.
        
        Args:
            task_name: Name of task to create (uses default if not provided)
            task_config: Configuration for the task
            task_id: Optional specific task ID
            
        Returns:
            ResetResult with initial observation and info
            
        Raises:
            ValueError: If no task name provided and no default set
            KeyError: If task name not found in registry
        """
        # Determine task to create
        task_name = task_name or self.default_task
        if not task_name:
            raise ValueError("No task name provided and no default task set")
        
        # Merge configs
        config = {**self.task_config, **(task_config or {})}
        
        # Validate config
        is_valid, error_msg = TaskValidator.validate_task_config(config)
        if not is_valid:
            raise ValueError(f"Invalid task config: {error_msg}")
        
        # Create task
        self.current_task = TaskRegistry.create_task(
            task_name=task_name,
            task_id=task_id,
            config=config
        )
        
        # Reset state manager with task info
        task_info = self.current_task.get_task_info()
        self.state_manager.reset(task_info)
        
        # Get initial observation from task
        initial_obs = self.current_task.get_initial_observation()
        
        self._episode_active = True
        
        return ResetResult(
            observation=initial_obs,
            info={
                "task_id": self.current_task.task_id,
                "task_type": self.current_task.task_type,
                "episode_id": self.state_manager.episode_id,
                "max_steps": self.max_steps_per_episode,
            }
        )
    
    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to execute
            
        Returns:
            StepResult containing observation, reward, done flag, and info
            
        Raises:
            RuntimeError: If called before reset() or after episode ended
        """
        if not self._episode_active:
            raise RuntimeError("Cannot step: environment not active. Call reset() first.")
        
        if not self.current_task:
            raise RuntimeError("No active task. Call reset() first.")
        
        # Validate action
        is_valid, error_msg = TaskValidator.validate_action(self.current_task, action)
        if not is_valid:
            # Return error observation
            observation = Observation(
                observation_type=ObservationType.ERROR,
                content=error_msg,
                success=False,
                error_message=error_msg
            )
            reward = Reward(
                value=-1.0,
                reward_type="penalty",
                reason=f"Invalid action: {error_msg}"
            )
            return StepResult(
                observation=observation,
                reward=reward,
                done=True,
                info={"error": error_msg}
            )
        
        # Execute action in task
        observation = self.current_task.execute_action(action)
        
        # Calculate reward
        reward = self.current_task.calculate_reward(action, observation)
        
        # Update task state
        self.current_task.steps_taken += 1
        self.current_task.update_status()
        
        # Record step in state manager
        self.state_manager.record_step(action, observation, reward)
        
        # Update task info in state
        updated_task_info = self.current_task.get_task_info()
        self.state_manager.update_task_info(updated_task_info)
        
        # Check if episode is done
        done = self._check_done()
        
        if done:
            self._episode_active = False
            self.state_manager.set_terminal(True)
        
        # Build info dict
        info = {
            "step": self.state_manager.step_count,
            "task_status": self.current_task.status.value,
            "task_progress": updated_task_info.progress,
            "total_reward": self.state_manager.total_reward,
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
    
    def state(self) -> EnvironmentState:
        """
        Get the current state of the environment.
        
        Returns:
            EnvironmentState object with complete current state
            
        Raises:
            RuntimeError: If called before reset()
        """
        if not self.current_task:
            raise RuntimeError("No active task. Call reset() first.")
        
        return self.state_manager.get_state()
    
    def _check_done(self) -> bool:
        """
        Check if the current episode is done.
        
        Returns:
            True if episode should end
        """
        if not self.current_task:
            return True
        
        # Check task completion/failure
        if self.current_task.is_complete() or self.current_task.is_failed():
            return True
        
        # Check max steps
        if self.state_manager.step_count >= self.max_steps_per_episode:
            return True
        
        # Check task-specific max steps
        if self.current_task.max_steps and \
           self.current_task.steps_taken >= self.current_task.max_steps:
            return True
        
        return False
    
    def render(self) -> str:
        """
        Render the current environment state as a string.
        
        Returns:
            Human-readable string representation of the state
        """
        if not self.current_task:
            return "Environment not initialized. Call reset() first."
        
        state = self.state_manager.get_state()
        task_info = state.task_info
        
        lines = [
            "=" * 60,
            f"OpenEnv Environment - Episode {state.episode_id[:8]}",
            "=" * 60,
            f"Task: {task_info.task_type} ({task_info.task_id[:8]})",
            f"Description: {task_info.description}",
            f"Status: {task_info.status.value}",
            f"Progress: {task_info.progress:.1f}%",
            "-" * 60,
            f"Step: {state.step_count} / {self.max_steps_per_episode}",
            f"Total Reward: {state.total_reward:.2f}",
            f"Terminal: {state.is_terminal}",
            "-" * 60,
            f"Workspace Files: {len(state.workspace.get('files', {}))}",
            f"Workspace Variables: {len(state.workspace.get('variables', {}))}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def close(self):
        """Clean up environment resources."""
        self._episode_active = False
        self.current_task = None
    
    @property
    def is_active(self) -> bool:
        """Check if environment is currently active."""
        return self._episode_active
    
    def get_available_tasks(self) -> list[str]:
        """Get list of available task types."""
        return TaskRegistry.list_tasks()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current episode."""
        stats = self.state_manager.get_statistics()
        if self.current_task:
            stats["task_type"] = self.current_task.task_type
            stats["task_status"] = self.current_task.status.value
        return stats
