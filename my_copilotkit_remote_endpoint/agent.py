# agent.py
from typing import Dict, Any
from langgraph.graph import Graph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List, Union
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolExecutor
import json
import os
from tools.intelsearch import search_inteleos_async
import asyncio
from config.endpoints import ENDPOINTS, Environment
from pydantic import BaseModel
from utils.logger import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv('ENVIRONMENT', 'LOCAL')
env = ENVIRONMENT.upper()
if env not in Environment.__members__:
    raise ValueError(f"Invalid environment: {env}.")
selected_env = Environment[env]
endpoints = ENDPOINTS[selected_env]
logger.info(f"Selected environment: {selected_env}")
logger.info(f"Endpoints: {endpoints}")


# Define state schema using Pydantic
class AgentState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    config: Dict
    context: Dict


# Convert the intelsearch tool to OpenAI tool format
intelsearch_tool_schema = convert_to_openai_tool(search_inteleos_async)
logger.info(f"Intelsearch tool schema: {intelsearch_tool_schema}")
# Initialize the model with tools
model = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o-mini",
    streaming=False
)
logger.info("Model initialized successfully")

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class Inteleos AI customer service assistant dedicated to providing exceptional support for Inteleos customers.
Questions about Inteleos? Use the available intelsearch tool to search inteleos.org for information.
Always respond with accurate Inteleos information in a friendly, professional, and empathetic manner."""),
    ("human", "{input}")
])
logger.info(f"Agent prompt: {AGENT_PROMPT}")


def should_continue(state: Dict[str, Any]) -> str:
    """Determine if the agent should continue or finish"""
    if not state.get("messages"):
        return "end"

    last_message = state["messages"][-1]
    logger.info(f"Last message: {last_message}")
    if isinstance(last_message, ToolMessage):
        return "continue"

    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("tool_calls"):
        return "continue"

    logger.info("Ending conversation")
    return "end"


async def process_questions(state: Dict, config: Dict, context: Dict) -> Dict:
    """Core agent logic with proper state management."""
    messages = state.get("messages", [])

    # Create proper state context
    current_state = {
        "messages": messages,
        "config": config.get("copilotkit_config", {}),
        "context": context
    }

    if messages and isinstance(messages[-1], AIMessage):
        return current_state

    try:
        response = await model.ainvoke(
            messages,
            config=current_state["config"]
        )

        if tool_calls := response.additional_kwargs.get("tool_calls"):
            try:
                # Iterate through tool calls
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "intelsearch":
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        intelsearch_result = await search_inteleos_async(tool_args.get("query"))
                        config = current_state["config"]
                        messages.append(response)
                        messages.append(
                            ToolMessage(
                                name="intelsearch",
                                content=str(intelsearch_result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                        break
            except json.JSONDecodeError:
                messages.append(
                    AIMessage(content="Error: Invalid tool arguments format")
                )
            except Exception as e:
                messages.append(
                    AIMessage(content=f"Error executing intelsearch tool: {str(e)}")
                )
        else:
            if not messages or not isinstance(messages[-1], AIMessage):
                messages.append(response)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        messages.append(
            AIMessage(content=f"An unexpected error occurred: {str(e)}")
        )

    return {
        "messages": messages,
        "config": current_state["config"],
        "context": context
    }
logger.info("Processed questions")
# Build the graph
workflow = StateGraph(AgentState)
logger.info("Graph built")
# Add state management
node_name = "questions_agent_node"
workflow.add_node(node_name, process_questions)
workflow.set_entry_point(node_name)
workflow.add_conditional_edges(
    node_name,
    should_continue,
    {
        "continue": node_name,
        "end": END
    }
)
logger.info("Conditional edges added")
logger.info("Workflow compiled")
# Compile the workflow
graph_agent = workflow.compile()
logger.info("Graph agent compiled")
