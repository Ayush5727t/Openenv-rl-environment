"""
Gradio App for OpenEnv Environment

This app provides a web interface for the OpenEnv environment,
suitable for deployment on Hugging Face Spaces.
"""

import gradio as gr
import json
from typing import Dict, List, Tuple
from src.environment import OpenEnvEnvironment
from models.models import Action, ActionType
import tasks  # Register tasks


# Global environment instance
env = None
history_log = []


def initialize_environment(task_name: str, max_steps: int = 100) -> str:
    """Initialize the environment with a specific task."""
    global env, history_log
    
    try:
        env = OpenEnvEnvironment(max_steps_per_episode=max_steps)
        
        # Task configurations
        task_configs = {
            "file_management": {
                "required_files": ["output.txt", "data.csv"],
                "required_content": {"output.txt": "result"}
            },
            "text_processing": {
                "input_text": "Hello World! This is a sample text for processing.",
                "operations": [
                    {"type": "count_words", "target": "9"},
                    {"type": "uppercase"}
                ]
            },
            "data_processing": {
                "data": [
                    {"id": 1, "name": "Alice", "score": 85},
                    {"id": 2, "name": "Bob", "score": 92},
                    {"id": 3, "name": "Charlie", "score": 78},
                    {"id": 4, "name": "David", "score": 88},
                ],
                "operations": [
                    {"type": "filter", "field": "score", "operator": ">", "value": 80},
                    {"type": "aggregate", "field": "score", "agg_type": "average"}
                ]
            }
        }
        
        config = task_configs.get(task_name, {})
        result = env.reset(task_name=task_name, task_config=config)
        
        history_log = []
        history_log.append(f"Environment initialized with task: {task_name}")
        history_log.append(f"Initial observation: {result.observation.content[:200]}...")
        
        return f"✓ Environment initialized!\n\nTask: {task_name}\nEpisode ID: {result.info['episode_id'][:8]}...\n\n{result.observation.content}"
        
    except Exception as e:
        return f"❌ Error initializing environment: {str(e)}"


def execute_action(
    action_type: str,
    param1_key: str = "",
    param1_value: str = "",
    param2_key: str = "",
    param2_value: str = "",
    param3_key: str = "",
    param3_value: str = ""
) -> Tuple[str, str, str]:
    """Execute an action in the environment."""
    global env, history_log
    
    if env is None or not env.is_active:
        return "❌ Environment not initialized. Please initialize first.", "", ""
    
    try:
        # Build parameters
        parameters = {}
        if param1_key and param1_value:
            # Try to parse as number
            try:
                parameters[param1_key] = int(param1_value)
            except ValueError:
                try:
                    parameters[param1_key] = float(param1_value)
                except ValueError:
                    parameters[param1_key] = param1_value
        
        if param2_key and param2_value:
            try:
                parameters[param2_key] = int(param2_value)
            except ValueError:
                try:
                    parameters[param2_key] = float(param2_value)
                except ValueError:
                    parameters[param2_key] = param2_value
        
        if param3_key and param3_value:
            try:
                parameters[param3_key] = int(param3_value)
            except ValueError:
                try:
                    parameters[param3_key] = float(param3_value)
                except ValueError:
                    parameters[param3_key] = param3_value
        
        # Create action
        action = Action(
            action_type=ActionType(action_type),
            parameters=parameters
        )
        
        # Execute step
        result = env.step(action)
        
        # Log to history
        history_log.append(f"\nAction: {action_type}")
        history_log.append(f"Parameters: {parameters}")
        history_log.append(f"Observation: {result.observation.content}")
        history_log.append(f"Reward: {result.reward.value} ({result.reward.reason})")
        history_log.append(f"Done: {result.done}")
        
        # Build output
        output = f"{'✓' if result.observation.success else '❌'} Action executed\n\n"
        output += f"Observation: {result.observation.content}\n\n"
        output += f"Reward: {result.reward.value} ({result.reward.reason})\n"
        output += f"Episode done: {result.done}\n"
        
        if result.done:
            output += f"\n🎉 Episode completed!\n"
        
        # Get state
        state = env.state()
        
        # Build state info
        state_info = f"Step: {state.step_count}\n"
        state_info += f"Total Reward: {state.total_reward:.2f}\n"
        state_info += f"Task Status: {state.task_info.status.value}\n"
        state_info += f"Progress: {state.task_info.progress:.1f}%\n"
        
        # Build history
        history_text = "\n".join(history_log[-20:])  # Last 20 entries
        
        return output, state_info, history_text
        
    except Exception as e:
        return f"❌ Error executing action: {str(e)}", "", ""


def get_current_state() -> str:
    """Get the current environment state."""
    global env
    
    if env is None:
        return "Environment not initialized"
    
    try:
        state = env.state()
        
        output = "=== Current Environment State ===\n\n"
        output += f"Episode ID: {state.episode_id[:8]}...\n"
        output += f"Step Count: {state.step_count}\n"
        output += f"Total Reward: {state.total_reward:.2f}\n"
        output += f"Terminal: {state.is_terminal}\n\n"
        
        output += f"=== Task Info ===\n"
        output += f"Task Type: {state.task_info.task_type}\n"
        output += f"Status: {state.task_info.status.value}\n"
        output += f"Progress: {state.task_info.progress:.1f}%\n"
        output += f"Description: {state.task_info.description}\n\n"
        
        output += f"=== Workspace ===\n"
        output += f"Files: {len(state.workspace.get('files', {}))}\n"
        output += f"Variables: {len(state.workspace.get('variables', {}))}\n"
        
        return output
        
    except Exception as e:
        return f"Error getting state: {str(e)}"


def get_available_tasks() -> str:
    """Get list of available tasks."""
    tasks_list = tasks.TaskRegistry.list_tasks()
    return "\n".join(f"- {task}" for task in tasks_list)


# Create Gradio interface
with gr.Blocks(title="OpenEnv Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 OpenEnv Environment
    
    A complete, real-world environment for AI agents to learn from through the standard step() / reset() / state() API.
    
    ## Available Tasks:
    - **file_management**: Create and manage files
    - **text_processing**: Process and transform text
    - **data_processing**: Filter, sort, and analyze data
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Initialize Environment")
            task_dropdown = gr.Dropdown(
                choices=["file_management", "text_processing", "data_processing"],
                value="file_management",
                label="Select Task"
            )
            max_steps_slider = gr.Slider(
                minimum=10,
                maximum=200,
                value=100,
                step=10,
                label="Max Steps"
            )
            init_btn = gr.Button("Initialize Environment", variant="primary")
            init_output = gr.Textbox(label="Initialization Result", lines=10)
            
        with gr.Column(scale=1):
            gr.Markdown("### 2️⃣ Execute Actions")
            action_type_dropdown = gr.Dropdown(
                choices=[
                    "file_write",
                    "file_read",
                    "file_list",
                    "text_edit",
                    "data_process",
                    "task_complete"
                ],
                label="Action Type"
            )
            
            with gr.Row():
                param1_key = gr.Textbox(label="Parameter 1 Key", placeholder="e.g., filename")
                param1_value = gr.Textbox(label="Parameter 1 Value", placeholder="e.g., output.txt")
            
            with gr.Row():
                param2_key = gr.Textbox(label="Parameter 2 Key", placeholder="e.g., content")
                param2_value = gr.Textbox(label="Parameter 2 Value", placeholder="e.g., result")
            
            with gr.Row():
                param3_key = gr.Textbox(label="Parameter 3 Key", placeholder="e.g., operation")
                param3_value = gr.Textbox(label="Parameter 3 Value", placeholder="e.g., filter")
            
            execute_btn = gr.Button("Execute Action", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Action Result")
            action_output = gr.Textbox(label="Result", lines=8)
        
        with gr.Column():
            gr.Markdown("### 📈 Current State")
            state_output = gr.Textbox(label="State", lines=8)
    
    with gr.Row():
        gr.Markdown("### 📜 History")
        history_output = gr.Textbox(label="Action History", lines=10)
    
    with gr.Row():
        state_btn = gr.Button("Get Full State")
        full_state_output = gr.Textbox(label="Complete State", lines=15)
    
    # Connect buttons
    init_btn.click(
        fn=initialize_environment,
        inputs=[task_dropdown, max_steps_slider],
        outputs=[init_output]
    )
    
    execute_btn.click(
        fn=execute_action,
        inputs=[
            action_type_dropdown,
            param1_key, param1_value,
            param2_key, param2_value,
            param3_key, param3_value
        ],
        outputs=[action_output, state_output, history_output]
    )
    
    state_btn.click(
        fn=get_current_state,
        outputs=[full_state_output]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Example Actions:
    
    **File Management:**
    - Action: `file_write` | Param1: `filename` = `output.txt` | Param2: `content` = `result here`
    
    **Text Processing:**
    - Action: `text_edit` | Param1: `operation` = `count` | Param2: `count_type` = `words`
    
    **Data Processing:**
    - Action: `data_process` | Param1: `operation` = `filter` | Param2: `field` = `score` | Param3: `operator` = `>`
    """)


if __name__ == "__main__":
    demo.launch()
