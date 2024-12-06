# tools/nutrition.py
from typing import Any, Dict, Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os
import asyncio
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


class NutritionInput(BaseModel):
    """Schema for nutrition request parameters"""
    query: str = Field(
        ...,
        description="Food item name or query to retrieve nutritional information."
    )
    config: Optional[Dict] = Field(
        default=None,
        description="Configuration options including API keys and settings"
    )


async def get_nutrition_info_async(query: str, config: Optional[Dict] = None) -> str:
    """
    Async implementation of nutrition information retrieval using Tavily Search.
    """
    config = config or {}
    api_key = config.get('api_key') or TAVILY_API_KEY
    if not api_key:
        raise ValueError("Nutrition (Tavily) API key is required")

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query)
        return format_nutrition_response(response, query)
    except Exception as e:
        return f"Unexpected error retrieving nutrition info: {str(e)}"


def get_nutrition_info(query: str, config: Optional[Dict] = None) -> str:
    """
    Synchronous wrapper for the async nutrition function.
    """
    return asyncio.run(get_nutrition_info_async(query, config))


def format_nutrition_response(data: Dict[str, Any], query: str) -> str:
    """Format the Tavily API search response into a human-readable string"""
    results = data.get("results", [])
    if not results:
        return f"No information found for '{query}'."

    formatted_results = []
    for i, result in enumerate(results[:3], start=1):
        title = result.get("title", "No Title")
        content = result.get("content", "No Content")
        formatted_results.append(
            f"Result {i}:\n"
            f"Title: {title}\n"
            f"Content Snippet: {content[:200]}...\n"
        )

    return (
        f"Search results for '{query}':\n" +
        "\n".join(formatted_results)
    )


# Create the tool using StructuredTool with the sync wrapper
nutrition_tool = StructuredTool.from_function(
    func=get_nutrition_info,
    name="get_nutrition_info",
    description="Retrieve search results related to a given query using Tavily.",
    args_schema=NutritionInput
)
