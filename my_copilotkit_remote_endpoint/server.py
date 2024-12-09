# server.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, LangGraphAgent
import uvicorn
import os
from fastapi.responses import StreamingResponse
from my_copilotkit_remote_endpoint.agent import graph_agent
import json
from fastapi.responses import JSONResponse
from my_copilotkit_remote_endpoint.utils.logger import logging
# from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS')

app = FastAPI()
logger.info("FastAPI app created")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

# allowed_origins = ALLOWED_ORIGINS.split(",") or ["*"]

if ALLOWED_ORIGINS:
    allowed_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
else:
    allowed_origins = ["*"]  # Fallback to wildcard if env is empty or not set
logger.warning("ALLOWED_ORIGINS not set. Using wildcard origin ('*').")

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/copilotkit_remote/info")
async def get_info():
    logger.info("GET request received")

    async def event_generator():
        # Create a JSON-encoded message
        message = json.dumps({"message": "GET request succeeded"})
        yield f"data: {message}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Define your backend action
async def fetch_name_for_user_id(userId: str):
    # Replace with your database logic
    logger.info(f"fetch_name_for_user_id called with userId: {userId}")
    return {"name": "User_" + userId}


# Initialize the CopilotKit SDK with the agent configuration
sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="weather_agent",
            description="Agent that answers questions about the weather",
            agent=graph_agent
        )
    ]
)

# Add the CopilotKit endpoint to your FastAPI app
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")
logger.info("CopilotKit endpoint added to FastAPI app")


def main():
    """Run the uvicorn server."""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


logger.info("Uvicorn server started")

if __name__ == "__main__":
    main()
logger.info("Server is running on port 8000")
