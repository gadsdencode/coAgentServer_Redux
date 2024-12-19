# agent_customer_service.py
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
from my_copilotkit_remote_endpoint.tools.order_tracker import order_status_tool
from pydantic import BaseModel
from langgraph.checkpoint import InMemoryCheckpointer

# Environment setup with proper validation
ENVIRONMENT = os.getenv('ENVIRONMENT', 'LOCAL')
env = ENVIRONMENT.upper()
if env not in Environment.__members__:
    raise ValueError(f"Invalid environment: {env}.")
selected_env = Environment[env]
endpoints = ENDPOINTS[selected_env]


# Define state schema using Pydantic
class AgentState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    config: Dict
    context: Dict


# Convert order tracking tool to OpenAI format
order_tool_schema = convert_to_openai_tool(order_status_tool.func)

# Initialize model with tools and proper configuration
model = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o-mini",
    streaming=False
).bind(tools=[order_tool_schema])

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class customer service assistant dedicated to providing exceptional support.
Use the available order_status_tool to provide accurate order tracking information when requested.
Always respond in a friendly, professional, and empathetic manner.
If an order is not found or there's an error, explain the issue clearly and offer alternative assistance."""),
    ("human", "{input}")
])


def should_continue(state: Dict[str, Any]) -> str:
    """Determine if the agent should continue processing"""
    if not state.get("messages"):
        return "end"

    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        return "continue"

    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("tool_calls"):
        return "continue"

    return "end"


async def process_customer_support(state: Dict, config: Dict, context: Dict) -> Dict:
    """Core agent logic with comprehensive state management and error handling"""
    messages = state.get("messages", [])

    # Initialize proper state context
    current_state = {
        "messages": messages,
        "config": config.get("copilotkit_config", {}),
        "context": context
    }

    if messages and isinstance(messages[-1], AIMessage):
        return current_state

    try:
        # Process with proper config context
        response = await model.ainvoke(
            messages,
            config=current_state["config"]
        )

        if tool_calls := response.additional_kwargs.get("tool_calls"):
            try:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "get_order_status":
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        order_result = await order_status_tool.func(
                            tool_args.get("order_id"),
                            config=current_state["config"]
                        )
                        messages.append(response)
                        messages.append(
                            ToolMessage(
                                name="get_order_status",
                                content=str(order_result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                        break
            except json.JSONDecodeError:
                messages.append(
                    AIMessage(content="Error: Invalid order ID format provided")
                )
            except Exception as e:
                messages.append(
                    AIMessage(content=f"Error checking order status: {str(e)}")
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

    # Return complete state with all context
    return {
        "messages": messages,
        "config": current_state["config"],
        "context": context
    }

# Build the graph
workflow = StateGraph(AgentState)

# Add state management
node_name = "customer_support_agent_node"
workflow.add_node(node_name, process_customer_support)
workflow.set_entry_point(node_name)
workflow.add_conditional_edges(
    node_name,
    should_continue,
    {
        "continue": node_name,
        "end": END
    }
)

# Compile the workflow with an in-memory checkpointer
graph_agent = workflow.compile(checkpointer=InMemoryCheckpointer())
