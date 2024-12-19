from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from copilotkit import CopilotKitSDK, LangGraphAgent
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Callable, Optional
from fastapi.middleware.base import BaseHTTPMiddleware
from my_copilotkit_remote_endpoint.utils.logger import setup_logger
import traceback
import uvicorn
import os

# Import agents with corrected import for customer_support_graph_agent
from my_copilotkit_remote_endpoint.agent import graph_agent
from my_copilotkit_remote_endpoint.agent_nutrition import graph_agent as nutrition_graph_agent
from my_copilotkit_remote_endpoint.agent_customer_support import graph_agent as customer_support_graph_agent

logger = setup_logger("copilotkit-server")

# Environment variable validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize FastAPI app
app = FastAPI()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(request)}"
        logger.info(f"[{request_id}] Started {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            logger.info(f"[{request_id}] Completed {response.status_code} in {response.headers.get('X-Process-Time', '0.00')}s")
            return response
        except Exception as e:
            logger.error(f"[{request_id}] Error processing request: {str(e)}\n{traceback.format_exc()}")
            raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://web-dev-461a.up.railway.app",
        "https://www.web-dev-461a.up.railway.app",
        "https://ai-customer-support-nine-eta.vercel.app",
        "https://www.ai-customer-support-nine-eta.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Origin", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type"],
    max_age=3600
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Initialize the SDK with agents
sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="inteleos_agent",
            description="Agent for handling Inteleos related queries",
            agent=graph_agent
        ),
        LangGraphAgent(
            name="nutrition_agent",
            description="Agent for handling nutrition related queries",
            agent=nutrition_graph_agent
        ),
        LangGraphAgent(
            name="customer_support_agent",
            description="Agent for handling customer support queries",
            agent=customer_support_graph_agent
        )
    ]
)

# Add the CopilotKit endpoint to your FastAPI app
from copilotkit.integrations.fastapi import add_fastapi_endpoint
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")

@app.post("/copilotkit_remote/assistants/search")
async def search_assistants(request: Request):
    request_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"[{request_id}] Search assistants endpoint called")

    try:
        body = await request.json()
        query = body.get("query", "").lower()
        logger.info(f"[{request_id}] Processing search query: {query}")

        available_agents = [
            {
                "id": agent.name,
                "name": agent.name.replace("_", " ").title(),
                "description": agent.description
            }
            for agent in sdk.agents
        ]

        if query:
            available_agents = [
                agent for agent in available_agents
                if query in agent["name"].lower() or query in agent["description"].lower()
            ]
            logger.info(f"[{request_id}] Found {len(available_agents)} matching agents")

        return JSONResponse(content={
            "assistants": available_agents,
            "total": len(available_agents)
        })

    except json.JSONDecodeError:
        logger.error(f"[{request_id}] Invalid JSON in request body")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON in request body"}
        )
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )


@app.get("/copilotkit_remote")
async def get_copilotkit_remote():
    """Endpoint for CopilotKit to fetch available actions"""
    logger.info("Remote endpoint GET called")
    return JSONResponse(content={
        "actions": [
            {
                "name": "inteleos_agent",
                "description": "Get information about Inteleos.org",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question about Inteleos.org"
                        }
                    }
                }
            },
            {
                "name": "nutrition_agent",
                "description": "Get nutrition information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Food item or nutrition query"
                        }
                    }
                }
            },
            {
                "name": "customer_support_agent",
                "description": "Get customer support",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Customer support query"
                        }
                    }
                }
            }
        ]
    })


@app.get("/copilotkit_remote/info")
async def get_copilotkit_info():
    """Endpoint for CopilotKit to fetch available actions"""
    logger.info("Info endpoint called")
    return JSONResponse(content={
        "actions": [
            {
                "name": "inteleos_agent",
                "description": "Get information about Inteleos.org",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question about Inteleos.org"
                        }
                    }
                }
            },
            {
                "name": "nutrition_agent",
                "description": "Get nutrition information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Food item or nutrition query"
                        }
                    }
                }
            },
            {
                "name": "customer_support_agent",
                "description": "Get customer support",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Customer support query"
                        }
                    }
                }
            }
        ]
    })


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server operation"""
    logger.info("Test endpoint called")
    return JSONResponse(content={"status": "ok", "message": "Server is running"})


@app.get("/")
async def root():
    """Root endpoint for basic health check"""
    logger.info("Root endpoint called")
    return JSONResponse(content={"status": "ok", "service": "copilotkit-server"})


@app.post("/copilotkit_remote/info")
async def post_copilotkit_info(request: Request):
    data = await request.json()
    logger.info(f"Received POST request: {data}")
    # Process data and return a response
    return JSONResponse(content={"status": "POST accepted"})

def main():
    """Run the uvicorn server."""
    logger.info("Starting uvicorn server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
