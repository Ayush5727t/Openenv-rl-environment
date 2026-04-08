"""
Data Processing Task - Teach agents to process and analyze data.

This task requires agents to perform data operations like filtering,
sorting, aggregating, and transforming structured data.
"""

from typing import Any, Dict, List
from models.models import (
    Action, Observation, Reward, TaskInfo,
    ActionType, ObservationType, RewardType
)
from tasks.base_task import Task, TaskRegistry


@TaskRegistry.register("data_processing")
class DataProcessingTask(Task):
    """
    Task where agent must process structured data.
    
    Example scenarios:
    - Filter data based on criteria
    - Sort data by specific fields
    - Calculate aggregates (sum, average, etc.)
    - Transform data format
    """
    
    def _initialize_task(self):
        """Initialize data processing task."""
        # Sample data
        default_data = [
            {"id": 1, "name": "Alice", "score": 85, "category": "A"},
            {"id": 2, "name": "Bob", "score": 72, "category": "B"},
            {"id": 3, "name": "Charlie", "score": 90, "category": "A"},
            {"id": 4, "name": "David", "score": 65, "category": "C"},
            {"id": 5, "name": "Eve", "score": 88, "category": "A"},
        ]
        
        self.task_data = {
            "original_data": self.config.get("data", default_data),
            "current_data": self.config.get("data", default_data).copy(),
            "required_operations": self.config.get("operations", [
                {"type": "filter", "field": "score", "operator": ">", "value": 80}
            ]),
            "completed_operations": [],
            "results": {},
        }
    
    def get_task_info(self) -> TaskInfo:
        """Get current task information."""
        total_ops = len(self.task_data["required_operations"])
        completed = len(self.task_data["completed_operations"])
        progress = (completed / total_ops * 100) if total_ops > 0 else 0
        
        return TaskInfo(
            task_id=self.task_id,
            task_type=self.task_type,
            description=f"Process data using {total_ops} operations",
            status=self.status,
            progress=progress,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            metadata={
                "data_size": len(self.task_data["current_data"]),
                "operations_required": total_ops,
            }
        )
    
    def get_initial_observation(self) -> Observation:
        """Get initial task description."""
        data = self.task_data["original_data"]
        ops = self.task_data["required_operations"]
        
        description = "Data Processing Task\n\n"
        description += f"Dataset: {len(data)} records\n"
        description += f"Sample record: {data[0] if data else 'No data'}\n\n"
        
        description += "Required operations:\n"
        for i, op in enumerate(ops, 1):
            op_desc = f"{i}. {op.get('type', 'unknown')}"
            if 'field' in op:
                op_desc += f" on field '{op['field']}'"
            if 'operator' in op:
                op_desc += f" {op['operator']} {op.get('value', '')}"
            description += f"  {op_desc}\n"
        
        description += "\nUse data_process action to manipulate the data."
        
        return Observation(
            observation_type=ObservationType.TEXT,
            content=description,
            metadata={
                "task_type": "data_processing",
                "data_size": len(data)
            }
        )
    
    def execute_action(self, action: Action) -> Observation:
        """Execute data processing action."""
        if action.action_type != ActionType.DATA_PROCESS:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unsupported action: {action.action_type}",
                success=False
            )
        
        params = action.parameters
        operation = params.get("operation")
        
        if operation == "filter":
            return self._do_filter(params)
        elif operation == "sort":
            return self._do_sort(params)
        elif operation == "aggregate":
            return self._do_aggregate(params)
        elif operation == "transform":
            return self._do_transform(params)
        elif operation == "view":
            return self._do_view(params)
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unknown operation: {operation}",
                success=False
            )
    
    def _do_filter(self, params: Dict[str, Any]) -> Observation:
        """Filter data based on criteria."""
        field = params.get("field")
        operator = params.get("operator", "==")
        value = params.get("value")
        
        if not field:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing 'field' parameter",
                success=False
            )
        
        original_count = len(self.task_data["current_data"])
        
        # Apply filter
        filtered = []
        for record in self.task_data["current_data"]:
            if field not in record:
                continue
            
            record_value = record[field]
            matches = False
            
            if operator == "==":
                matches = record_value == value
            elif operator == ">":
                matches = record_value > value
            elif operator == "<":
                matches = record_value < value
            elif operator == ">=":
                matches = record_value >= value
            elif operator == "<=":
                matches = record_value <= value
            elif operator == "!=":
                matches = record_value != value
            
            if matches:
                filtered.append(record)
        
        self.task_data["current_data"] = filtered
        
        # Check if this completes a requirement
        self._check_operation_completion("filter", {
            "field": field,
            "operator": operator,
            "value": value
        })
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content=f"Filtered data: {original_count} -> {len(filtered)} records",
            metadata={
                "operation": "filter",
                "original_count": original_count,
                "filtered_count": len(filtered)
            }
        )
    
    def _do_sort(self, params: Dict[str, Any]) -> Observation:
        """Sort data by field."""
        field = params.get("field")
        reverse = params.get("reverse", False)
        
        if not field:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing 'field' parameter",
                success=False
            )
        
        try:
            self.task_data["current_data"].sort(
                key=lambda x: x.get(field, 0),
                reverse=reverse
            )
            
            self._check_operation_completion("sort", {"field": field})
            
            return Observation(
                observation_type=ObservationType.SUCCESS,
                content=f"Sorted data by '{field}' ({'descending' if reverse else 'ascending'})",
                metadata={"operation": "sort", "field": field}
            )
        except Exception as e:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Sort failed: {str(e)}",
                success=False
            )
    
    def _do_aggregate(self, params: Dict[str, Any]) -> Observation:
        """Calculate aggregates on data."""
        agg_type = params.get("agg_type", "count")
        field = params.get("field")
        
        data = self.task_data["current_data"]
        
        if agg_type == "count":
            result = len(data)
        elif agg_type == "sum":
            if not field:
                return Observation(
                    observation_type=ObservationType.ERROR,
                    content="Missing 'field' for sum",
                    success=False
                )
            result = sum(record.get(field, 0) for record in data)
        elif agg_type == "average":
            if not field:
                return Observation(
                    observation_type=ObservationType.ERROR,
                    content="Missing 'field' for average",
                    success=False
                )
            values = [record.get(field, 0) for record in data]
            result = sum(values) / len(values) if values else 0
        elif agg_type == "max":
            if not field:
                return Observation(
                    observation_type=ObservationType.ERROR,
                    content="Missing 'field' for max",
                    success=False
                )
            result = max((record.get(field, 0) for record in data), default=0)
        elif agg_type == "min":
            if not field:
                return Observation(
                    observation_type=ObservationType.ERROR,
                    content="Missing 'field' for min",
                    success=False
                )
            result = min((record.get(field, 0) for record in data), default=0)
        else:
            return Observation(
                observation_type=ObservationType.ERROR,
                content=f"Unknown aggregate type: {agg_type}",
                success=False
            )
        
        self.task_data["results"][f"{agg_type}_{field or 'all'}"] = result
        
        self._check_operation_completion("aggregate", {
            "agg_type": agg_type,
            "field": field
        })
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content=f"{agg_type.capitalize()} result: {result}",
            metadata={"operation": "aggregate", "result": result}
        )
    
    def _do_transform(self, params: Dict[str, Any]) -> Observation:
        """Transform data fields."""
        field = params.get("field")
        transform_type = params.get("transform_type", "uppercase")
        
        if not field:
            return Observation(
                observation_type=ObservationType.ERROR,
                content="Missing 'field' parameter",
                success=False
            )
        
        for record in self.task_data["current_data"]:
            if field in record:
                if transform_type == "uppercase":
                    record[field] = str(record[field]).upper()
                elif transform_type == "lowercase":
                    record[field] = str(record[field]).lower()
        
        self._check_operation_completion("transform", {
            "field": field,
            "transform_type": transform_type
        })
        
        return Observation(
            observation_type=ObservationType.SUCCESS,
            content=f"Transformed field '{field}' using {transform_type}",
            metadata={"operation": "transform"}
        )
    
    def _do_view(self, params: Dict[str, Any]) -> Observation:
        """View current data state."""
        limit = params.get("limit", 5)
        data_sample = self.task_data["current_data"][:limit]
        
        return Observation(
            observation_type=ObservationType.TEXT,
            content=f"Current data ({len(self.task_data['current_data'])} records): {data_sample}",
            metadata={"data_size": len(self.task_data["current_data"])}
        )
    
    def _check_operation_completion(self, op_type: str, params: Dict[str, Any]):
        """Check if an operation requirement is met."""
        for i, required_op in enumerate(self.task_data["required_operations"]):
            if i in self.task_data["completed_operations"]:
                continue
            
            if required_op.get("type") == op_type:
                # Check if parameters match
                match = True
                for key in ["field", "operator", "value", "agg_type"]:
                    if key in required_op:
                        if params.get(key) != required_op[key]:
                            match = False
                            break
                
                if match:
                    self.task_data["completed_operations"].append(i)
    
    def calculate_reward(self, action: Action, observation: Observation) -> Reward:
        """Calculate reward for action."""
        if not observation.success:
            return Reward(
                value=-0.1,
                reward_type=RewardType.PENALTY,
                reason="Action failed"
            )
        
        # Reward for completing operations
        params = action.parameters
        if params.get("operation") in ["filter", "sort", "aggregate", "transform"]:
            # Check if we just completed a requirement
            total_ops = len(self.task_data["required_operations"])
            completed = len(self.task_data["completed_operations"])
            
            if completed == total_ops:
                return Reward(
                    value=5.0,
                    reward_type=RewardType.TASK_COMPLETION,
                    reason="All operations completed!"
                )
            elif completed > 0:
                return Reward(
                    value=2.0,
                    reward_type=RewardType.TASK_PROGRESS,
                    reason="Completed a required operation"
                )
        
        return Reward(
            value=0.1,
            reward_type=RewardType.EFFICIENCY,
            reason="Action completed successfully"
        )
    
    def is_complete(self) -> bool:
        """Check if all operations are complete."""
        total_ops = len(self.task_data["required_operations"])
        completed = len(self.task_data["completed_operations"])
        return completed >= total_ops
    
    def is_failed(self) -> bool:
        """This task doesn't have failure conditions."""
        return False
