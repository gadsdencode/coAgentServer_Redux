# agent_nutrition.py
from typing import Dict, Any, List, Union
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
import json
import os
import asyncio
from my_copilotkit_remote_endpoint.config.endpoints import ENDPOINTS, Environment
from my_copilotkit_remote_endpoint.tools.nutrition import nutrition_tool
from pydantic import BaseModel
from langgraph.checkpoint import SimpleDictCheckpointer

# Environment setup
ENVIRONMENT = os.getenv('ENVIRONMENT', 'LOCAL')
env = ENVIRONMENT.upper()
if env not in Environment.__members__:
    raise ValueError(f"Invalid environment: {env}.")
selected_env = Environment[env]
endpoints = ENDPOINTS[selected_env]

# Convert nutrition tool to OpenAI format
nutrition_tool_schema = convert_to_openai_tool(nutrition_tool.func)

# Initialize model with tools
model = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o-mini",
    streaming=False
).bind(tools=[nutrition_tool_schema])

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class nutrition and fitness assistant dedicated to providing exceptional guidance.
Use the available nutrition_tool to provide accurate nutritional information when requested.
Always respond in a friendly, professional, and empathetic manner."""),
    ("human", "{input}")
])


def should_continue(state: Dict[str, Any]) -> str:
    """Determine if the agent should continue or finish"""
    if not state.get("messages"):
        return "end"

    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        return "continue"

    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("tool_calls"):
        return "continue"

    return "end"


async def process_nutrition(state: Dict, config: Dict, context: Dict) -> Dict:
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
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "get_nutrition_info":
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        nutrition_result = await nutrition_tool.func(
                            tool_args.get("query"),
                            config=current_state["config"]
                        )
                        messages.append(response)
                        messages.append(
                            ToolMessage(
                                name="get_nutrition_info",
                                content=str(nutrition_result),
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
                    AIMessage(content=f"Error executing nutrition tool: {str(e)}")
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


# Define state schema using Pydantic
class AgentState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    config: Dict
    context: Dict


# Build the graph
workflow = StateGraph(AgentState)

# Add state management
node_name = "nutrition_agent_node"
workflow.add_node(node_name, process_nutrition)
workflow.set_entry_point(node_name)
workflow.add_conditional_edges(
    node_name,
    should_continue,
    {
        "continue": node_name,
        "end": END
    }
)

# Compile the workflow with a simple dict checkpointer
graph_agent = workflow.compile(checkpointer={"type": "dict"})
