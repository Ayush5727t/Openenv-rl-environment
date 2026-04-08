"""
Tests for OpenEnv Environment Core Components
"""

import pytest
from datetime import datetime
from models.models import (
    Action, Observation, Reward, TaskInfo, EnvironmentState,
    ActionType, ObservationType, RewardType, TaskStatus
)


class TestModels:
    """Test Pydantic models."""
    
    def test_action_creation(self):
        """Test creating an Action."""
        action = Action(
            action_type=ActionType.FILE_WRITE,
            parameters={"filename": "test.txt", "content": "hello"}
        )
        assert action.action_type == ActionType.FILE_WRITE
        assert action.parameters["filename"] == "test.txt"
        assert isinstance(action.timestamp, datetime)
    
    def test_observation_creation(self):
        """Test creating an Observation."""
        obs = Observation(
            observation_type=ObservationType.TEXT,
            content="Test observation",
            success=True
        )
        assert obs.observation_type == ObservationType.TEXT
        assert obs.content == "Test observation"
        assert obs.success is True
    
    def test_reward_creation(self):
        """Test creating a Reward."""
        reward = Reward(
            value=1.0,
            reward_type=RewardType.TASK_PROGRESS,
            reason="Task completed"
        )
        assert reward.value == 1.0
        assert reward.reward_type == RewardType.TASK_PROGRESS
    
    def test_task_info_validation(self):
        """Test TaskInfo progress validation."""
        task_info = TaskInfo(
            task_id="test-1",
            task_type="test",
            description="Test task",
            progress=50.0
        )
        assert task_info.progress == 50.0
        
        # Test clamping
        task_info2 = TaskInfo(
            task_id="test-2",
            task_type="test",
            description="Test task",
            progress=150.0  # Should be clamped to 100
        )
        assert task_info2.progress == 100.0
    
    def test_environment_state_history(self):
        """Test EnvironmentState history management."""
        task_info = TaskInfo(
            task_id="test",
            task_type="test",
            description="Test"
        )
        
        state = EnvironmentState(
            task_info=task_info,
            episode_id="episode-1"
        )
        
        # Add some history
        action = Action(action_type=ActionType.FILE_READ)
        obs = Observation(observation_type=ObservationType.TEXT, content="test")
        reward = Reward(value=1.0, reward_type=RewardType.TASK_PROGRESS, reason="test")
        
        state.add_to_history(action, obs, reward)
        assert len(state.history) == 1
        assert state.history[0]["step"] == 0


class TestEnvironment:
    """Test OpenEnv Environment."""
    
    def test_environment_initialization(self):
        """Test environment can be initialized."""
        from src.environment import OpenEnvEnvironment
        
        env = OpenEnvEnvironment(default_task="file_management")
        assert env.default_task == "file_management"
        assert not env.is_active
    
    def test_environment_reset(self):
        """Test environment reset."""
        from src.environment import OpenEnvEnvironment
        import tasks  # Import to register tasks
        
        env = OpenEnvEnvironment()
        result = env.reset(task_name="file_management", task_config={
            "required_files": ["test.txt"]
        })
        
        assert result.observation is not None
        assert "task_id" in result.info
        assert env.is_active
    
    def test_environment_step(self):
        """Test environment step."""
        from src.environment import OpenEnvEnvironment
        import tasks
        
        env = OpenEnvEnvironment()
        env.reset(task_name="file_management", task_config={
            "required_files": ["test.txt"]
        })
        
        action = Action(
            action_type=ActionType.FILE_WRITE,
            parameters={"filename": "test.txt", "content": "hello"}
        )
        
        result = env.step(action)
        assert result.observation is not None
        assert result.reward is not None
        assert isinstance(result.done, bool)
    
    def test_environment_state(self):
        """Test getting environment state."""
        from src.environment import OpenEnvEnvironment
        import tasks
        
        env = OpenEnvEnvironment()
        env.reset(task_name="file_management")
        
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.episode_id is not None
        assert state.step_count >= 0


class TestStateManager:
    """Test StateManager."""
    
    def test_state_manager_initialization(self):
        """Test state manager initialization."""
        from src.state_manager import StateManager
        
        sm = StateManager()
        assert sm.step_count == 0
        assert sm.total_reward == 0.0
        assert len(sm.history) == 0
    
    def test_state_manager_reset(self):
        """Test state manager reset."""
        from src.state_manager import StateManager
        
        sm = StateManager()
        task_info = TaskInfo(
            task_id="test",
            task_type="test",
            description="Test"
        )
        
        state = sm.reset(task_info)
        assert state.episode_id is not None
        assert state.task_info.task_id == "test"
    
    def test_state_manager_record_step(self):
        """Test recording a step."""
        from src.state_manager import StateManager
        
        sm = StateManager()
        task_info = TaskInfo(
            task_id="test",
            task_type="test",
            description="Test"
        )
        sm.reset(task_info)
        
        action = Action(action_type=ActionType.FILE_READ)
        obs = Observation(observation_type=ObservationType.TEXT, content="test")
        reward = Reward(value=1.0, reward_type=RewardType.TASK_PROGRESS, reason="test")
        
        sm.record_step(action, obs, reward)
        
        assert sm.step_count == 1
        assert sm.total_reward == 1.0
        assert len(sm.history) == 1
    
    def test_workspace_file_operations(self):
        """Test workspace file operations."""
        from src.state_manager import StateManager
        
        sm = StateManager()
        
        sm.add_file("test.txt", "hello")
        assert sm.get_file("test.txt") == "hello"
        assert "test.txt" in sm.list_files()
        
        sm.delete_file("test.txt")
        assert sm.get_file("test.txt") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
