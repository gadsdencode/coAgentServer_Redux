# config/endpoints.py

from enum import Enum


class Environment(Enum):
    LOCAL = 'LOCAL'
    PRODUCTION = 'PRODUCTION'


ENDPOINTS = {
    Environment.LOCAL: {
        'BASE': 'http://localhost:8000/copilotkit_remote',
        'ACTIONS': '/info',
        'STREAM': '/stream',
        'HEALTH': '/health',
    },
    Environment.PRODUCTION: {
        'BASE': 'https://web-production-7cd0b.up.railway.app/copilotkit_remote',
        'ACTIONS': '/info',
        'STREAM': '/stream',
        'HEALTH': '/health',
    },
}
