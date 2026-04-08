"""
Tests for Task implementations
"""

import pytest
from models.models import Action, ActionType
from tasks.base_task import TaskRegistry
import tasks  # Import to register tasks


class TestFileManagementTask:
    """Test FileManagementTask."""
    
    def test_task_creation(self):
        """Test creating a file management task."""
        task = TaskRegistry.create_task(
            "file_management",
            config={"required_files": ["test.txt", "output.txt"]}
        )
        
        assert task.task_type == "FileManagementTask"
        assert len(task.task_data["required_files"]) == 2
    
    def test_initial_observation(self):
        """Test initial observation."""
        task = TaskRegistry.create_task(
            "file_management",
            config={"required_files": ["test.txt"]}
        )
        
        obs = task.get_initial_observation()
        assert obs.success
        assert "test.txt" in obs.content
    
    def test_file_write_action(self):
        """Test file write action."""
        task = TaskRegistry.create_task(
            "file_management",
            config={"required_files": ["test.txt"]}
        )
        
        action = Action(
            action_type=ActionType.FILE_WRITE,
            parameters={"filename": "test.txt", "content": "hello"}
        )
        
        obs = task.execute_action(action)
        assert obs.success
        assert "test.txt" in task.task_data["created_files"]
    
    def test_task_completion(self):
        """Test task completion detection."""
        task = TaskRegistry.create_task(
            "file_management",
            config={"required_files": ["test.txt"]}
        )
        
        assert not task.is_complete()
        
        action = Action(
            action_type=ActionType.FILE_WRITE,
            parameters={"filename": "test.txt", "content": "hello"}
        )
        task.execute_action(action)
        
        assert task.is_complete()
    
    def test_forbidden_files(self):
        """Test forbidden file detection."""
        task = TaskRegistry.create_task(
            "file_management",
            config={
                "required_files": ["test.txt"],
                "forbidden_files": ["bad.txt"]
            }
        )
        
        action = Action(
            action_type=ActionType.FILE_WRITE,
            parameters={"filename": "bad.txt", "content": "forbidden"}
        )
        
        obs = task.execute_action(action)
        assert not obs.success
        assert task.is_failed()


class TestTextProcessingTask:
    """Test TextProcessingTask."""
    
    def test_task_creation(self):
        """Test creating a text processing task."""
        task = TaskRegistry.create_task(
            "text_processing",
            config={
                "input_text": "Hello World",
                "operations": [{"type": "count_words"}]
            }
        )
        
        assert task.task_type == "TextProcessingTask"
        assert task.task_data["input_text"] == "Hello World"
    
    def test_text_replacement(self):
        """Test text replacement operation."""
        task = TaskRegistry.create_task(
            "text_processing",
            config={
                "input_text": "Hello World",
                "operations": [{"type": "replace", "target": "World"}]
            }
        )
        
        action = Action(
            action_type=ActionType.TEXT_EDIT,
            parameters={
                "operation": "replace",
                "pattern": "World",
                "replacement": "Earth"
            }
        )
        
        obs = task.execute_action(action)
        assert obs.success
        assert "Earth" in task.task_data["current_text"]
    
    def test_word_count(self):
        """Test word counting."""
        task = TaskRegistry.create_task(
            "text_processing",
            config={
                "input_text": "Hello World Test",
                "operations": [{"type": "count_words", "target": "3"}]
            }
        )
        
        action = Action(
            action_type=ActionType.TEXT_EDIT,
            parameters={
                "operation": "count",
                "count_type": "words"
            }
        )
        
        obs = task.execute_action(action)
        assert obs.success
        assert task.is_complete()


class TestDataProcessingTask:
    """Test DataProcessingTask."""
    
    def test_task_creation(self):
        """Test creating a data processing task."""
        data = [
            {"id": 1, "score": 85},
            {"id": 2, "score": 90}
        ]
        
        task = TaskRegistry.create_task(
            "data_processing",
            config={
                "data": data,
                "operations": [{"type": "filter", "field": "score", "operator": ">", "value": 80}]
            }
        )
        
        assert len(task.task_data["original_data"]) == 2
    
    def test_filter_operation(self):
        """Test filtering data."""
        data = [
            {"id": 1, "score": 85},
            {"id": 2, "score": 75},
            {"id": 3, "score": 90}
        ]
        
        task = TaskRegistry.create_task(
            "data_processing",
            config={
                "data": data,
                "operations": [{"type": "filter", "field": "score", "operator": ">", "value": 80}]
            }
        )
        
        action = Action(
            action_type=ActionType.DATA_PROCESS,
            parameters={
                "operation": "filter",
                "field": "score",
                "operator": ">",
                "value": 80
            }
        )
        
        obs = task.execute_action(action)
        assert obs.success
        assert len(task.task_data["current_data"]) == 2  # Only 85 and 90
        assert task.is_complete()
    
    def test_aggregate_operation(self):
        """Test aggregate operations."""
        data = [
            {"id": 1, "score": 80},
            {"id": 2, "score": 90},
            {"id": 3, "score": 70}
        ]
        
        task = TaskRegistry.create_task(
            "data_processing",
            config={
                "data": data,
                "operations": [{"type": "aggregate", "field": "score", "agg_type": "average"}]
            }
        )
        
        action = Action(
            action_type=ActionType.DATA_PROCESS,
            parameters={
                "operation": "aggregate",
                "field": "score",
                "agg_type": "average"
            }
        )
        
        obs = task.execute_action(action)
        assert obs.success
        assert "80" in obs.content  # Average is 80


class TestTaskRegistry:
    """Test TaskRegistry."""
    
    def test_list_tasks(self):
        """Test listing registered tasks."""
        tasks_list = TaskRegistry.list_tasks()
        
        assert "file_management" in tasks_list
        assert "text_processing" in tasks_list
        assert "data_processing" in tasks_list
    
    def test_create_unknown_task(self):
        """Test creating unknown task raises error."""
        with pytest.raises(KeyError):
            TaskRegistry.create_task("unknown_task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
