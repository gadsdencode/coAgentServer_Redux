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

API_KEY = os.getenv('ALLOWED_ORIGINS')

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

allowed_origins = API_KEY.split(",") or ["*"]
if not allowed_origins or allowed_origins == [""]:
    allowed_origins = ["*"]  # Fallback to wildcard if env is empty

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
    async def event_generator():
        # Create a JSON-encoded message
        message = json.dumps({"message": "GET request succeeded"})
        yield f"data: {message}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Define your backend action
async def fetch_name_for_user_id(userId: str):
    # Replace with your database logic
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


def main():
    """Run the uvicorn server."""
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
