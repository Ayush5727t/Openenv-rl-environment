"""Configuration for OpenEnv Environment Package"""

__version__ = "1.0.0"
__author__ = "OpenEnv Team"
__license__ = "MIT"

# Package metadata
PACKAGE_NAME = "openenv"
DESCRIPTION = "A complete environment for AI agents with multiple task types"
URL = "https://github.com/yourusername/openenv"

# Environment defaults
DEFAULT_MAX_STEPS = 1000
DEFAULT_TASK = "file_management"

# Available task types
AVAILABLE_TASKS = [
    "file_management",
    "text_processing",
    "data_processing",
]

# Action types
ACTION_TYPES = [
    "file_read",
    "file_write",
    "file_delete",
    "file_list",
    "api_call",
    "data_process",
    "text_edit",
    "command_exec",
    "task_complete",
]
