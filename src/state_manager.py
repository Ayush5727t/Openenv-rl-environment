"""
State Manager - Handles environment state persistence and history.

This module manages the environment's state, including history tracking,
workspace management, and state persistence.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
from models.models import (
    Action, Observation, Reward, EnvironmentState,
    TaskInfo, TaskStatus
)
import uuid


class StateManager:
    """
    Manages environment state including history and workspace.
    
    Responsibilities:
    - Track action/observation/reward history
    - Manage workspace state (files, variables, etc.)
    - Provide state snapshots
    - Handle state persistence (optional)
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        """
        Initialize the state manager.
        
        Args:
            persistence_dir: Optional directory for saving state to disk
        """
        self.persistence_dir = persistence_dir
        if persistence_dir:
            persistence_dir.mkdir(parents=True, exist_ok=True)
        
        self._reset_state()
    
    def _reset_state(self):
        """Reset all state to initial values."""
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.history: List[Dict[str, Any]] = []
        self.workspace: Dict[str, Any] = {
            "files": {},  # filename -> content
            "variables": {},  # variable_name -> value
            "outputs": [],  # list of outputs/logs
        }
        self.is_terminal = False
        self.current_task_info: Optional[TaskInfo] = None
    
    def reset(self, task_info: TaskInfo) -> EnvironmentState:
        """
        Reset the state manager for a new episode.
        
        Args:
            task_info: Information about the new task
            
        Returns:
            Initial environment state
        """
        self._reset_state()
        self.current_task_info = task_info
        return self.get_state()
    
    def record_step(self, action: Action, observation: Observation, reward: Reward):
        """
        Record a step in the environment.
        
        Args:
            action: Action taken
            observation: Observation received
            reward: Reward received
        """
        self.step_count += 1
        self.total_reward += reward.value
        
        step_record = {
            "step": self.step_count,
            "action": action.model_dump(),
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.history.append(step_record)
        
        # Optionally persist to disk
        if self.persistence_dir:
            self._persist_step(step_record)
    
    def update_workspace(self, updates: Dict[str, Any]):
        """
        Update the workspace state.
        
        Args:
            updates: Dictionary of updates to apply to workspace
        """
        for key, value in updates.items():
            if key in self.workspace and isinstance(self.workspace[key], dict):
                self.workspace[key].update(value)
            else:
                self.workspace[key] = value
    
    def add_file(self, filename: str, content: str):
        """Add or update a file in the workspace."""
        self.workspace["files"][filename] = content
    
    def get_file(self, filename: str) -> Optional[str]:
        """Get file content from workspace."""
        return self.workspace["files"].get(filename)
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file from workspace. Returns True if file existed."""
        return self.workspace["files"].pop(filename, None) is not None
    
    def list_files(self) -> List[str]:
        """Get list of files in workspace."""
        return list(self.workspace["files"].keys())
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the workspace."""
        self.workspace["variables"][name] = value
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get a variable from the workspace."""
        return self.workspace["variables"].get(name)
    
    def add_output(self, output: str):
        """Add an output/log entry."""
        self.workspace["outputs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "content": output
        })
    
    def get_state(self) -> EnvironmentState:
        """
        Get the current environment state.
        
        Returns:
            EnvironmentState object with complete current state
        """
        if not self.current_task_info:
            raise RuntimeError("No task info set. Call reset() first.")
        
        return EnvironmentState(
            task_info=self.current_task_info,
            episode_id=self.episode_id,
            step_count=self.step_count,
            total_reward=self.total_reward,
            history=self.history[-100:],  # Last 100 steps
            workspace=self._get_workspace_snapshot(),
            is_terminal=self.is_terminal,
            timestamp=datetime.utcnow()
        )
    
    def _get_workspace_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of workspace state for serialization."""
        return {
            "files": dict(self.workspace["files"]),
            "variables": dict(self.workspace["variables"]),
            "outputs": list(self.workspace["outputs"][-50:]),  # Last 50 outputs
        }
    
    def set_terminal(self, is_terminal: bool):
        """Mark the episode as terminal (ended)."""
        self.is_terminal = is_terminal
    
    def update_task_info(self, task_info: TaskInfo):
        """Update the current task information."""
        self.current_task_info = task_info
    
    def _persist_step(self, step_record: Dict[str, Any]):
        """Persist a step record to disk."""
        if not self.persistence_dir:
            return
        
        episode_dir = self.persistence_dir / self.episode_id
        episode_dir.mkdir(exist_ok=True)
        
        # Append to episode history file
        history_file = episode_dir / "history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(step_record) + '\n')
    
    def save_state(self, filename: Optional[str] = None):
        """
        Save the current state to disk.
        
        Args:
            filename: Optional filename (default: state_<episode_id>.json)
        """
        if not self.persistence_dir:
            raise RuntimeError("No persistence directory configured")
        
        state = self.get_state()
        filename = filename or f"state_{self.episode_id}.json"
        filepath = self.persistence_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(state.model_dump_json(indent=2))
    
    def load_state(self, filename: str) -> EnvironmentState:
        """
        Load state from disk.
        
        Args:
            filename: Name of the state file to load
            
        Returns:
            Loaded environment state
        """
        if not self.persistence_dir:
            raise RuntimeError("No persistence directory configured")
        
        filepath = self.persistence_dir / filename
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        state = EnvironmentState(**state_data)
        
        # Restore internal state
        self.episode_id = state.episode_id
        self.step_count = state.step_count
        self.total_reward = state.total_reward
        self.history = state.history
        self.workspace = state.workspace
        self.is_terminal = state.is_terminal
        self.current_task_info = state.task_info
        
        return state
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n most recent history entries.
        
        Args:
            n: Number of recent entries to return
            
        Returns:
            List of recent history entries
        """
        return self.history[-n:] if self.history else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / self.step_count if self.step_count > 0 else 0.0,
            "is_terminal": self.is_terminal,
            "files_count": len(self.workspace["files"]),
            "variables_count": len(self.workspace["variables"]),
        }
