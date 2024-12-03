# setup_agent.py

import asyncio
import os
from langgraph_sdk import get_client
from my_copilotkit_remote_endpoint.config.endpoints import ENDPOINTS, Environment
from dotenv import load_dotenv


load_dotenv()


async def setup_agent():
    # Determine the environment; default to PRODUCTION if not specified
    env = os.getenv('ENVIRONMENT', 'PRODUCTION').upper()
    if env not in Environment.__members__:
        raise ValueError(f"Invalid environment: {env}. Choose from {[e.name for e in Environment]}.")

    selected_env = Environment[env]
    endpoints = ENDPOINTS[selected_env]

    # Construct the full BASE URL
    base_url = endpoints['BASE']

    # Initialize the client with the BASE URL
    client = get_client(url=base_url)

    # Select an assistant that is not configured
    assistants = await client.assistants.search()
    assistant = next((a for a in assistants if not a.get("config")), None)
    if not assistant:
        raise RuntimeError("No unconfigured assistant found.")

    # Retrieve configuration schemas
    schemas = await client.assistants.get_schemas(
        assistant_id=assistant["assistant_id"]
    )

    # Display config schema (for reference)
    print("Configuration Schema:", schemas["config_schema"])

    # Create a configured assistant using OpenAI
    openai_assistant = await client.assistants.create(
        "agent",
        config={"configurable": {"model_name": "openai"}}
    )

    print("Configured Assistant:", openai_assistant)

    # Optionally, verify the configuration by initiating a thread
    thread = await client.threads.create()
    input_data = {"messages": [{"role": "user", "content": "Who made you?"}]}

    async for event in client.runs.stream(
        thread["thread_id"],
        openai_assistant["assistant_id"],
        input=input_data,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")

# Entry point
if __name__ == "__main__":
    asyncio.run(setup_agent())
