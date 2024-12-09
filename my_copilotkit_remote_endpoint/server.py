from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from copilotkit import CopilotKitSDK, LangGraphAgent
from fastapi.responses import JSONResponse
from langsmith import Client
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

# Import agents
from my_copilotkit_remote_endpoint.agent import graph_agent
from my_copilotkit_remote_endpoint.agent_nutrition import graph_agent as nutrition_graph_agent
from my_copilotkit_remote_endpoint.agent_customer_support import graph_agent as customer_support_graph_agent

logger = setup_logger("copilotkit-server")

# Load environment variables
# load_dotenv()

# Environment variable validation
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY environment variable is not set")

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS')

# Initialize LangSmith client
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# Configure LangSmith environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-internal-kayak-74"

# Initialize FastAPI app
app = FastAPI()


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


class LangSmithTracingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, project_name: str, exclude_paths: list[str] = None):
        super().__init__(app)
        self.project_name = project_name
        self.exclude_paths = exclude_paths or ["/test", "/", "/copilotkit_remote/info"]

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        trace_id = f"copilotkit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(request)}"
        logger.info(f"Starting trace for request: {trace_id}")

        # Replace the trace context manager with run_and_trace
        try:
            response = await langsmith_client.run_and_trace(
                project_name=self.project_name,
                name=trace_id,
                tags=["copilotkit", request.method.lower()],
                run_type="chain",
            )(call_next)(request)

            return response
        except Exception as e:
            logger.error(f"Tracing error: {str(e)}")
            # Fallback to normal request processing if tracing fails
            return await call_next(request)


class TracedCopilotKitSDK(CopilotKitSDK):
    async def _process_request(self, request: dict, trace: Optional[any] = None) -> dict:
        try:
            if trace:
                with trace.branch(run_name="process_request"):
                    response = await super()._process_request(request)
                    return response
            return await super()._process_request(request)
        except Exception as e:
            if trace:
                trace.on_chain_error(e)
            raise


# Initialize the SDK with agents
sdk = TracedCopilotKitSDK(
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

# Configure CORS
if ALLOWED_ORIGINS:
    allowed_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    LangSmithTracingMiddleware,
    project_name=os.getenv("LANGCHAIN_PROJECT", "224a7_1_3")
)


# Dependency for getting current trace
async def get_current_trace(request: Request):
    return getattr(request.state, "langsmith_trace", None)


def add_traced_fastapi_endpoint(app: FastAPI, sdk: TracedCopilotKitSDK, path: str):
    """Add CopilotKit endpoint with LangSmith tracing"""
    logger.info(f"Adding traced FastAPI endpoint at path: {path}")

    @app.post(path)
    async def copilotkit_endpoint(
        request: Request,
        trace: Optional[any] = Depends(get_current_trace)
    ):
        request_id = f"copilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[{request_id}] Processing request")

        body = await request.json()
        response = await sdk._process_request(body, trace)
        logger.info(f"[{request_id}] Request processed successfully")

        return response

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

# Add the traced endpoint
add_traced_fastapi_endpoint(app, sdk, "/copilotkit_remote")


def main():
    """Run the uvicorn server."""
    logger.info("Starting uvicorn server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
