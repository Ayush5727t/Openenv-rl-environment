"""
Example: Using the OpenEnv Environment

This script demonstrates how to use the OpenEnv environment
with different tasks.
"""

from src.environment import OpenEnvEnvironment
from models.models import Action, ActionType
import tasks  # Import to register tasks


def example_file_management():
    """Example: File Management Task"""
    print("=" * 60)
    print("Example 1: File Management Task")
    print("=" * 60)
    
    # Create environment
    env = OpenEnvEnvironment(max_steps_per_episode=50)
    
    # Reset with file management task
    result = env.reset(
        task_name="file_management",
        task_config={
            "required_files": ["output.txt", "data.csv"],
            "required_content": {"output.txt": "result"},
        }
    )
    
    print("\nInitial Observation:")
    print(result.observation.content)
    
    # Step 1: Create first file
    action1 = Action(
        action_type=ActionType.FILE_WRITE,
        parameters={"filename": "output.txt", "content": "Here is the result"}
    )
    step_result = env.step(action1)
    print(f"\nStep 1: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value} ({step_result.reward.reason})")
    
    # Step 2: Create second file
    action2 = Action(
        action_type=ActionType.FILE_WRITE,
        parameters={"filename": "data.csv", "content": "id,value\n1,100"}
    )
    step_result = env.step(action2)
    print(f"\nStep 2: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value} ({step_result.reward.reason})")
    print(f"Task Complete: {step_result.done}")
    
    # Get final state
    state = env.state()
    print(f"\nFinal Statistics:")
    print(f"  Steps: {state.step_count}")
    print(f"  Total Reward: {state.total_reward}")
    print(f"  Task Status: {state.task_info.status.value}")
    print(f"  Progress: {state.task_info.progress}%")


def example_text_processing():
    """Example: Text Processing Task"""
    print("\n\n" + "=" * 60)
    print("Example 2: Text Processing Task")
    print("=" * 60)
    
    env = OpenEnvEnvironment()
    
    # Reset with text processing task
    result = env.reset(
        task_name="text_processing",
        task_config={
            "input_text": "Hello World! This is a test.",
            "operations": [
                {"type": "count_words", "target": "6"},
                {"type": "uppercase"}
            ]
        }
    )
    
    print("\nInitial Observation:")
    print(result.observation.content)
    
    # Step 1: Count words
    action1 = Action(
        action_type=ActionType.TEXT_EDIT,
        parameters={"operation": "count", "count_type": "words"}
    )
    step_result = env.step(action1)
    print(f"\nStep 1: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value}")
    
    # Step 2: Convert to uppercase
    action2 = Action(
        action_type=ActionType.TEXT_EDIT,
        parameters={"operation": "uppercase"}
    )
    step_result = env.step(action2)
    print(f"\nStep 2: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value}")
    print(f"Task Complete: {step_result.done}")


def example_data_processing():
    """Example: Data Processing Task"""
    print("\n\n" + "=" * 60)
    print("Example 3: Data Processing Task")
    print("=" * 60)
    
    env = OpenEnvEnvironment()
    
    # Sample dataset
    data = [
        {"id": 1, "name": "Alice", "score": 85, "grade": "B"},
        {"id": 2, "name": "Bob", "score": 92, "grade": "A"},
        {"id": 3, "name": "Charlie", "score": 78, "grade": "C"},
        {"id": 4, "name": "David", "score": 88, "grade": "B"},
    ]
    
    result = env.reset(
        task_name="data_processing",
        task_config={
            "data": data,
            "operations": [
                {"type": "filter", "field": "score", "operator": ">", "value": 80},
                {"type": "aggregate", "field": "score", "agg_type": "average"}
            ]
        }
    )
    
    print("\nInitial Observation:")
    print(result.observation.content)
    
    # Step 1: Filter high scores
    action1 = Action(
        action_type=ActionType.DATA_PROCESS,
        parameters={
            "operation": "filter",
            "field": "score",
            "operator": ">",
            "value": 80
        }
    )
    step_result = env.step(action1)
    print(f"\nStep 1: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value}")
    
    # Step 2: Calculate average
    action2 = Action(
        action_type=ActionType.DATA_PROCESS,
        parameters={
            "operation": "aggregate",
            "field": "score",
            "agg_type": "average"
        }
    )
    step_result = env.step(action2)
    print(f"\nStep 2: {step_result.observation.content}")
    print(f"Reward: {step_result.reward.value}")
    print(f"Task Complete: {step_result.done}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 60)
    print("# OpenEnv Environment Examples")
    print("#" * 60)
    
    try:
        example_file_management()
        example_text_processing()
        example_data_processing()
        
        print("\n\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
