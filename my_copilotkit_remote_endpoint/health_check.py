# health_check.py

import asyncio
import os
from langgraph_sdk import get_client
from my_copilotkit_remote_endpoint.config.endpoints import ENDPOINTS, Environment
# from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# load_dotenv(find_dotenv())

# Load environment variables
ENVIRONMENT = os.getenv('ENVIRONMENT', 'LOCAL')


async def health_check():
    # Determine the environment; default to PRODUCTION if not specified
    env = ENVIRONMENT.upper()
    if env not in Environment.__members__:
        raise ValueError(f"Invalid environment: {env}. Choose from {[e.name for e in Environment]}.")

    selected_env = Environment[env]
    endpoints = ENDPOINTS[selected_env]

    # Construct the HEALTH URL
    health_url = endpoints['BASE'] + endpoints['HEALTH']

    # Initialize the client (assuming get_client can accept full URLs for specific endpoints)
    client = get_client(url=health_url)

    try:
        response = await client.get_health()  # Replace with actual method to perform health check
        if response.status_code == 200:
            print("LangGraph service is healthy and operational.")
        else:
            print(f"Health check failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Health check encountered an error: {str(e)}")

# Entry point
if __name__ == "__main__":
    asyncio.run(health_check())
