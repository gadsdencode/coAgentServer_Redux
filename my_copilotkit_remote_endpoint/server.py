# /my_copilotkit_remote_endpoint/route.py

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from copilotkit import CopilotKitSDK, LangGraphAgent
from fastapi.responses import JSONResponse
from datetime import datetime
# from dotenv import load_dotenv
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uvicorn
import os
import json
from my_copilotkit_remote_endpoint.utils.logger import setup_logger
import traceback

# Import agents with corrected import for customer_support_graph_agent
from my_copilotkit_remote_endpoint.agent import graph_agent
from my_copilotkit_remote_endpoint.agent_nutrition import graph_agent as nutrition_graph_agent
from my_copilotkit_remote_endpoint.agent_customer_support import graph_agent as customer_support_graph_agent

logger = setup_logger("copilotkit-server")

# Load environment variables
# load_dotenv()

# Environment variable validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS')

# Initialize FastAPI app
app = FastAPI()

@app.middleware("http")
async def add_cors_headers_on_error(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Create a JSON response with the error
        error_response = JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )
        
        # Manually add CORS headers to error response
        error_response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        error_response.headers["Access-Control-Allow-Credentials"] = "true"
        error_response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        error_response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Origin, Authorization, X-Requested-With"
        
        return error_response

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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(request)}"
        logger.info(f"[{request_id}] Started {request.method} {request.url.path}")
        start_time = datetime.now()

        try:
            response = await call_next(request)
            process_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{request_id}] Completed {response.status_code} in {process_time:.2f}s")
            return response
        except Exception as e:
            logger.error(f"[{request_id}] Failed {request.method} {request.url.path}\n"
                        f"Error: {str(e)}\n"
                        f"Traceback: {traceback.format_exc()}")
            raise


class CopilotKitServerSDK(CopilotKitSDK):
    async def _process_request(self, request: dict) -> dict:
        try:
            response = await super().handle_request(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Initialize the SDK with agents
sdk = CopilotKitServerSDK(
    agents=[
        LangGraphAgent(
            name="inteleos_agent",
            description="Agent that answers questions about Inteleos",
            graph=graph_agent
        ),
        LangGraphAgent(
            name="nutrition_agent",
            description="Agent that provides nutrition and fitness guidance",
            graph=nutrition_graph_agent
        ),
        LangGraphAgent(
            name="customer_support_agent",
            description="Agent that provides world-class customer support",
            graph=customer_support_graph_agent
        )
    ]
)


# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Dependency for getting current trace (removed as tracing is no longer used)
# async def get_current_trace(request: Request):
#     return getattr(request.state, "langsmith_trace", None)


def add_fastapi_endpoint(app: FastAPI, sdk: CopilotKitServerSDK, path: str):
    """Add CopilotKit endpoint without LangSmith tracing"""
    logger.info(f"Adding FastAPI endpoint at path: {path}")

    @app.post(path)
    async def copilotkit_endpoint(request: Request):
        request_id = f"copilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            logger.info(f"[{request_id}] Processing request")
            body = await request.json()
            logger.debug(f"[{request_id}] Request body: {body}")
            
            response = await sdk._process_request(body)
            logger.debug(f"[{request_id}] Response: {response}")
            logger.info(f"[{request_id}] Request processed successfully")
            return response
        except Exception as e:
            logger.error(f"[{request_id}] Error processing request: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": str(e), "timestamp": datetime.utcnow().isoformat()}
            )

    return app


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


# Add the endpoint without LangSmith tracing
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")


def main():
    """Run the uvicorn server."""
    logger.info("Starting uvicorn server...")
    uvicorn.run("route:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
