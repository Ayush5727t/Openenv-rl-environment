"""
Text Processing Task - Teach agents to process and transform text data.

This task requires agents to perform text operations like searching,
replacing, formatting, and analyzing text according to requirements.
"""

from typing import Any, Dict, List
from models.models import (
    Action, Observation, Reward, TaskInfo,
    ActionType, ObservationType, RewardType
)
from tasks.base_task import Task, TaskRegistry
import re


@TaskRegistry.register("text_processing")
class TextProcessingTask(Task):
    """
    Task where agent must process text to meet specific requirements.
    
    Example scenarios:
    - Find and replace specific patterns
    - Extract information from text
    - Format text according to rules
    - Count occurrences of patterns
    """
    
    def _initialize_task(self):
        """Initialize text processing task."""
        # Get input text
        self.task_data = {
            "input_text": self.config.get("input_text", "Sample text for processing."),
            "operations": self.config.get("operations", []),  # List of required operations
            "current_text": self.config.get("input_text", "Sample text for processing."),
            "completed_operations": [],
            "operation_results": {},
        }
        
        # Default operations if none specified
        if not self.task_data["operations"]:
            self.task_data["operations"] = [
                {"type": "count_words", "target": "complete"}
            ]
    
    def get_task_info(self) -> TaskInfo:
        """Get current task information."""
        total_ops = len(self.task_data["operations"])
        completed = len(self.task_data["completed_operations"])
        progress = (completed / total_ops * 100) if total_ops > 0 else 0
        
        return TaskInfo(
            task_id=self.task_id,
            task_type=self.task_type,
            description=f"Process text using {total_ops} operations",
            status=self.status,
            progress=progress,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            metadata={
                "operations_required": total_ops,
                "operations_completed": completed,
            }
        )
    
    def get_initial_observation(self) -> Observation:
        """Get initial task description."""
        ops = self.task_data["operations"]
        text = self.task_data["input_text"]
        
        description = "Text Processing Task\n\n"
        description += f"Input text ({len(text)} chars):\n{text[:200]}\n"
        if len(text) > 200:
            description += "...\n"
        
        description += f"\nRequired operations:\n"
        for i, op in enumerate(ops, 1):
            description += f"  {i}. {op.get('type', 'unknown')}"
            if 'target' in op:
                description += f" (target: {op['target']})"
            description += "\n"
        
        description += "\nUse text_edit action to process the text."
        
        return Observation(
            observation_type=ObservationType.TEXT,
            content=description,
            metadata={
                "task_type": "text_processing",
                "text_length": len(text)
            }
        )
    
    def execute_action(self, action: Action) -> Observation:
        """Execute text processing action."""
        action_type = action.action_type
        params = action.parameters
        
        if action_type == ActionType.TEXT_EDIT:
            return self._handle_text_edit(params)
        elif action_type == ActionType.DATA_PROCESS:
            return self._handle_data_process(params)
        elif action_type == ActionType.TASK_COMPLETE:
            return self._handle_task_complete()
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unsupported action: {action_type}",
                success=False
            )
    
    def _handle_text_edit(self, params: Dict[str, Any]) -> Observation:
        """Handle text editing operations."""
        operation = params.get("operation")
        
        if operation == "replace":
            return self._do_replace(params)
        elif operation == "uppercase":
            return self._do_uppercase(params)
        elif operation == "lowercase":
            return self._do_lowercase(params)
        elif operation == "count":
            return self._do_count(params)
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unknown text operation: {operation}",
                success=False
            )
    
    def _do_replace(self, params: Dict[str, Any]) -> Observation:
        """Replace text pattern."""
        pattern = params.get("pattern", "")
        replacement = params.get("replacement", "")
        
        if not pattern:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing 'pattern' parameter",
                success=False
            )
        
        old_text = self.task_data["current_text"]
        new_text = old_text.replace(pattern, replacement)
        count = old_text.count(pattern)
        
        self.task_data["current_text"] = new_text
        
        # Check if this completes an operation
        self._check_operation_completion("replace", pattern)
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content=f"Replaced {count} occurrence(s) of '{pattern}' with '{replacement}'",
            metadata={
                "operation": "replace",
                "count": count,
                "new_length": len(new_text)
            }
        )
    
    def _do_uppercase(self, params: Dict[str, Any]) -> Observation:
        """Convert text to uppercase."""
        self.task_data["current_text"] = self.task_data["current_text"].upper()
        
        self._check_operation_completion("uppercase", None)
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content="Converted text to uppercase",
            metadata={"operation": "uppercase"}
        )
    
    def _do_lowercase(self, params: Dict[str, Any]) -> Observation:
        """Convert text to lowercase."""
        self.task_data["current_text"] = self.task_data["current_text"].lower()
        
        self._check_operation_completion("lowercase", None)
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content="Converted text to lowercase",
            metadata={"operation": "lowercase"}
        )
    
    def _do_count(self, params: Dict[str, Any]) -> Observation:
        """Count occurrences or words."""
        count_type = params.get("count_type", "words")
        
        text = self.task_data["current_text"]
        
        if count_type == "words":
            count = len(text.split())
            self.task_data["operation_results"]["word_count"] = count
            self._check_operation_completion("count_words", count)
            
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content=f"Word count: {count}",
                metadata={"count": count, "count_type": "words"}
            )
        
        elif count_type == "chars":
            count = len(text)
            self.task_data["operation_results"]["char_count"] = count
            self._check_operation_completion("count_chars", count)
            
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content=f"Character count: {count}",
                metadata={"count": count, "count_type": "chars"}
            )
        
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unknown count type: {count_type}",
                success=False
            )
    
    def _handle_data_process(self, params: Dict[str, Any]) -> Observation:
        """Handle data processing operations."""
        # Return current text state
        return Observation(
            observation_type=ObservationType.TEXT,
            content=self.task_data["current_text"],
            metadata={"length": len(self.task_data["current_text"])}
        )
    
    def _check_operation_completion(self, op_type: str, result: Any):
        """Check if an operation requirement is met."""
        for i, op in enumerate(self.task_data["operations"]):
            if i in self.task_data["completed_operations"]:
                continue
            
            if op.get("type") == op_type:
                # Check if target is met (if specified)
                if "target" in op:
                    if str(result) == str(op["target"]):
                        self.task_data["completed_operations"].append(i)
                else:
                    self.task_data["completed_operations"].append(i)
    
    def _handle_task_complete(self) -> Observation:
        """Handle task completion declaration."""
        if self.is_complete():
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content="Text processing task completed successfully!"
            )
        else:
            remaining = len(self.task_data["operations"]) - len(self.task_data["completed_operations"])
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Task not complete. {remaining} operations remaining.",
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
        
        # Check if operation was just completed
        prev_completed = len(self.task_data["completed_operations"])
        
        # Reward for completing an operation requirement
        if action.action_type == ActionType.TEXT_EDIT:
            if len(self.task_data["completed_operations"]) > prev_completed:
                return Reward(
                    value=2.0,
                    reward_type=RewardType.TASK_PROGRESS,
                    reason="Completed a required operation"
                )
        
        # Small reward for successful actions
        return Reward(
            value=0.1,
            reward_type=RewardType.EFFICIENCY,
            reason="Action completed successfully"
        )
    
    def is_complete(self) -> bool:
        """Check if all operations are complete."""
        total_ops = len(self.task_data["operations"])
        completed = len(self.task_data["completed_operations"])
        return completed >= total_ops
    
    def is_failed(self) -> bool:
        """Check if task has failed."""
        # This task doesn't have failure conditions beyond max steps
        return False
