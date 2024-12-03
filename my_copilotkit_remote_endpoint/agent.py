# agent.py
from typing import Dict, List, Tuple, Any
from langgraph.graph import Graph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
import json
import os
from my_copilotkit_remote_endpoint.tools.weather import get_weather_async
import asyncio
from functools import partial
from my_copilotkit_remote_endpoint.config.endpoints import ENDPOINTS, Environment

ENVIRONMENT = os.getenv('ENVIRONMENT', 'PRODUCTION')

# Select the environment; default to PRODUCTION
env = ENVIRONMENT.upper()
if env not in Environment.__members__:
    raise ValueError(f"Invalid environment: {env}. Choose from {[e.name for e in Environment]}.")

selected_env = Environment[env]
endpoints = ENDPOINTS[selected_env]

# Convert the StructuredTool to OpenAI tool format
weather_tool_schema = convert_to_openai_tool(get_weather_async)

# Initialize the model with tools
model = ChatOpenAI(
    temperature=0,
    model="gpt-4-1106-preview",
    streaming=True
).bind(tools=[weather_tool_schema])

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful weather assistant that can provide accurate weather information.
    Use the available get_weather tool to help answer user questions about the weather.
    Always provide clear, concise responses and handle errors gracefully.
    If the location is not found or there's an error, explain it clearly to the user."""),
    ("human", "{input}"),
])


def should_continue(state: Dict) -> str:
    """Determine if the agent should continue or finish"""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("tool_calls"):
        return "continue"
    return "end"


async def agent(state: Dict, config: Dict) -> Dict:
    """Core agent logic with robust error handling and async support, now leveraging LangGraph config"""
    messages = state["messages"]

    try:
        # Generate agent response using the configured model
        response = await model.ainvoke(messages, config=config)

        # Handle tool calling
        if tool_calls := response.additional_kwargs.get("tool_calls"):
            try:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "get_weather":
                        # Parse tool arguments
                        tool_args = json.loads(tool_call["function"]["arguments"])

                        # Execute weather tool asynchronously
                        weather_result = await get_weather_async(tool_args.get("query"))

                        # Add tool response to messages as ToolMessage
                        messages.append(response)
                        messages.append(
                            ToolMessage(
                                name="get_weather",
                                content=str(weather_result),
                                tool_call_id=tool_call["id"]
                            )
                        )

                # Generate final response after tool execution
                final_response = await model.ainvoke(messages, config=config)
                messages.append(final_response)

            except json.JSONDecodeError:
                messages.append(
                    AIMessage(content="Error: Invalid tool arguments format")
                )
            except Exception as e:
                messages.append(
                    AIMessage(content=f"Error executing weather tool: {str(e)}")
                )
        else:
            # Add regular response to messages
            messages.append(response)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        messages.append(
            AIMessage(content=f"An unexpected error occurred: {str(e)}")
        )

    return {"messages": messages}


# Initialize the checkpointer
checkpointer = MemorySaver()

# Build the graph
workflow = Graph()
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# Compile the graph with checkpointing
graph_agent = workflow.compile(checkpointer=checkpointer)
