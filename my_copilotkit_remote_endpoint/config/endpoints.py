# config/endpoints.py

from enum import Enum


class Environment(Enum):
    LOCAL = 'LOCAL'
    PRODUCTION = 'PRODUCTION'


ENDPOINTS = {
    Environment.LOCAL: {
        'BASE': 'http://localhost:8000/copilotkit_remote',
        'BASE_ASSISTANTS': 'http://localhost:8000',
        'ASSISTANTS': '/copilotkit_remote/assistants/search',
        'ACTIONS': '/copilotkit_remote/info',
        'STREAM': '/copilotkit_remote',
        'HEALTH': '/health',
        'TEST': '/test',  # Add test endpoint
        'ROOT': '/'       # Add root endpoint
    },
    Environment.PRODUCTION: {
        'BASE': 'https://web-dev-461a.up.railway.app/copilotkit_remote',
        'BASE_ASSISTANTS': 'https://web-dev-461a.up.railway.app',
        'ASSISTANTS': '/copilotkit_remote/assistants/search',
        'ACTIONS': '/copilotkit_remote/info',
        'STREAM': '/copilotkit_remote',
        'HEALTH': '/health',
    },
}
