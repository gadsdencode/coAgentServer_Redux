# tools/intelsearch.py
from typing import Any, Dict, Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os
import asyncio
from tavily import TavilyClient
from my_copilotkit_remote_endpoint.utils.logger import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


class IntelSearchInput(BaseModel):
    """Schema for searching inteleos.org"""
    query: str = Field(
        ...,
        description="Query to search for on inteleos.org"
    )
    config: Optional[Dict] = Field(
        default=None,
        description="Configuration options including API keys and settings"
    )


logger.info(f"IntelSearchInput: {IntelSearchInput}")


async def search_inteleos_async(query: str, config: Optional[Dict] = None) -> str:
    """
    Async search on inteleos.org using Tavily Search.
    """
    config = config or {}
    api_key = config.get('api_key') or TAVILY_API_KEY
    if not api_key:
        raise ValueError("Tavily API key is required for IntelSearch")

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(f"site:inteleos.org {query}")
        return format_inteleos_response(response, query)
    except Exception as e:
        return f"Error searching inteleos.org: {str(e)}"
logger.info(f"search_inteleos_async: {search_inteleos_async}")


def search_inteleos(query: str, config: Optional[Dict] = None) -> str:
    """Sync wrapper for the async inteleos search."""
    return asyncio.run(search_inteleos_async
                       (query, config))


logger.info(f"search_inteleos: {search_inteleos}")


def format_inteleos_response(data: Dict[str, Any], query: str) -> str:
    """Format the Tavily response for inteleos.org search"""
    results = data.get("results", [])
    if not results:
        return f"No information found on inteleos.org for '{query}'."
    logger.info(f"format_inteleos_response: {format_inteleos_response}")
    formatted_results = []
    for i, result in enumerate(results[:3], start=1):
        title = result.get("title", "No Title")
        content = result.get("content", "No Content")
        url = result.get("url", "No URL")
        snippet = content[:200].replace('\n', ' ')
        formatted_results.append(
            f"Result {i}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}...\n"
        )
    logger.info(f"formatted_results: {formatted_results}")
    return (
        f"Search results from inteleos.org for '{query}':\n" +
        "\n".join(formatted_results)
    )


logger.info(f"format_inteleos_response: {format_inteleos_response}")

intelsearch_tool = StructuredTool.from_function(
    func=search_inteleos,
    name="search_inteleos",
    description=(
        "Search for information exclusively on https://inteleos.org and "
        "related pages to answer user queries."
    ),
    args_schema=IntelSearchInput
)
logger.info(f"intelsearch_tool: {intelsearch_tool}")
