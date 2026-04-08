"""
File Management Task - Teach agents to manage files in a workspace.

This task requires the agent to perform file operations like creating,
reading, updating, and deleting files according to specific requirements.
"""

from typing import Any, Dict
from models.models import (
    Action, Observation, Reward, TaskInfo, TaskStatus,
    ActionType, ObservationType, RewardType
)
from tasks.base_task import Task, TaskRegistry


@TaskRegistry.register("file_management")
class FileManagementTask(Task):
    """
    Task where agent must manage files to meet specific requirements.
    
    Example scenarios:
    - Create a specific set of files
    - Organize files into correct structure
    - Clean up unnecessary files
    - Update file contents
    """
    
    def _initialize_task(self):
        """Initialize file management task."""
        # Task requirements
        self.task_data = {
            "required_files": self.config.get("required_files", ["output.txt"]),
            "required_content": self.config.get("required_content", {}),
            "forbidden_files": self.config.get("forbidden_files", []),
            "created_files": [],
            "completed_requirements": set(),
        }
        
        # Initial workspace files (if any)
        initial_files = self.config.get("initial_files", {})
        self.task_data["initial_files"] = initial_files
    
    def get_task_info(self) -> TaskInfo:
        """Get current task information."""
        total_requirements = len(self.task_data["required_files"])
        completed = len(self.task_data["completed_requirements"])
        progress = (completed / total_requirements * 100) if total_requirements > 0 else 0
        
        return TaskInfo(
            task_id=self.task_id,
            task_type=self.task_type,
            description=f"Create and manage {total_requirements} required files",
            status=self.status,
            progress=progress,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            metadata={
                "required_files": self.task_data["required_files"],
                "completed": list(self.task_data["completed_requirements"]),
            }
        )
    
    def get_initial_observation(self) -> Observation:
        """Get initial task description."""
        required = self.task_data["required_files"]
        content_reqs = self.task_data["required_content"]
        
        description = f"File Management Task\n\n"
        description += f"Required files to create:\n"
        for filename in required:
            description += f"  - {filename}"
            if filename in content_reqs:
                description += f" (must contain: '{content_reqs[filename]}')"
            description += "\n"
        
        if self.task_data["forbidden_files"]:
            description += f"\nDo not create these files: {', '.join(self.task_data['forbidden_files'])}\n"
        
        description += "\nUse file_write action to create files."
        
        return Observation(
            observation_type=ObservationType.TEXT,
            content=description,
            metadata={"task_type": "file_management"}
        )
    
    def execute_action(self, action: Action) -> Observation:
        """Execute file operation."""
        action_type = action.action_type
        params = action.parameters
        
        if action_type == ActionType.FILE_WRITE:
            return self._handle_file_write(params)
        elif action_type == ActionType.FILE_READ:
            return self._handle_file_read(params)
        elif action_type == ActionType.FILE_DELETE:
            return self._handle_file_delete(params)
        elif action_type == ActionType.FILE_LIST:
            return self._handle_file_list(params)
        elif action_type == ActionType.TASK_COMPLETE:
            return self._handle_task_complete()
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unsupported action: {action_type}",
                success=False,
                error_message=f"Action {action_type} not supported in file management task"
            )
    
    def _handle_file_write(self, params: Dict[str, Any]) -> Observation:
        """Handle file write action."""
        filename = params.get("filename")
        content = params.get("content", "")
        
        if not filename:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing required parameter: filename",
                success=False
            )
        
        # Check if forbidden
        if filename in self.task_data["forbidden_files"]:
            self.task_data["created_files"].append(filename)
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Cannot create forbidden file: {filename}",
                success=False
            )
        
        # Create/update file
        self.task_data["created_files"].append(filename)
        
        # Check if this satisfies a requirement
        if filename in self.task_data["required_files"]:
            required_content = self.task_data["required_content"].get(filename)
            if not required_content or required_content in content:
                self.task_data["completed_requirements"].add(filename)
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content=f"File '{filename}' created successfully",
            metadata={
                "filename": filename,
                "size": len(content)
            }
        )
    
    def _handle_file_read(self, params: Dict[str, Any]) -> Observation:
        """Handle file read action."""
        filename = params.get("filename")
        
        if not filename:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing required parameter: filename",
                success=False
            )
        
        # For simplicity, we don't actually store file contents in this task
        # Just report if file was created
        if filename in self.task_data["created_files"]:
            return Observation(
                observation_type=ObservationType.FILE_CONTENT,
                content=f"Content of {filename}",
                metadata={"filename": filename}
            )
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"File not found: {filename}",
                success=False
            )
    
    def _handle_file_delete(self, params: Dict[str, Any]) -> Observation:
        """Handle file delete action."""
        filename = params.get("filename")
        
        if filename in self.task_data["created_files"]:
            self.task_data["created_files"].remove(filename)
            self.task_data["completed_requirements"].discard(filename)
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content=f"File '{filename}' deleted successfully"
            )
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"File not found: {filename}",
                success=False
            )
    
    def _handle_file_list(self, params: Dict[str, Any]) -> Observation:
        """Handle file list action."""
        files = self.task_data["created_files"]
        return Observation(
            observation_type=ObservationType.FILE_LIST,
            content=files,
            metadata={"count": len(files)}
        )
    
    def _handle_task_complete(self) -> Observation:
        """Handle task completion declaration."""
        if self.is_complete():
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content="Task completed successfully!"
            )
        else:
            remaining = set(self.task_data["required_files"]) - self.task_data["completed_requirements"]
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Task not complete. Still need: {list(remaining)}",
                success=False
            )
    
    def calculate_reward(self, action: Action, observation: Observation) -> Reward:
        """Calculate reward for action."""
        if not observation.success:
            return Reward(
                value=-0.1,
                reward_type=RewardType.PENALTY,
                reason="Action failed"
            )
        
        # Reward for completing requirements
        if action.action_type == ActionType.FILE_WRITE:
            filename = action.parameters.get("filename")
            if filename in self.task_data["completed_requirements"]:
                return Reward(
                    value=1.0,
                    reward_type=RewardType.TASK_PROGRESS,
                    reason=f"Created required file: {filename}"
                )
        
        # Penalty for creating forbidden files
        if action.action_type == ActionType.FILE_WRITE:
            filename = action.parameters.get("filename")
            if filename in self.task_data["forbidden_files"]:
                return Reward(
                    value=-1.0,
                    reward_type=RewardType.PENALTY,
                    reason=f"Created forbidden file: {filename}"
                )
        
        # Small reward for successful actions
        return Reward(
            value=0.1,
            reward_type=RewardType.EFFICIENCY,
            reason="Action completed successfully"
        )
    
    def is_complete(self) -> bool:
        """Check if all requirements are met."""
        required = set(self.task_data["required_files"])
        completed = self.task_data["completed_requirements"]
        return required.issubset(completed)
    
    def is_failed(self) -> bool:
        """Check if task has failed."""
        # Fail if forbidden file was created
        forbidden = set(self.task_data["forbidden_files"])
        created = set(self.task_data["created_files"])
        return bool(forbidden.intersection(created))
