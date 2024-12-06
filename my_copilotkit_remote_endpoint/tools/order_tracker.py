# tools/order_tracker.py
from typing import Any, Dict
from langchain.tools import StructuredTool
import httpx
from pydantic import BaseModel, Field
import os
import asyncio

# Load environment variables if needed
ORDER_API_KEY = os.getenv('ORDER_API_KEY')


class OrderStatusInput(BaseModel):
    """Schema for order status request parameters"""
    order_id: str = Field(
        ...,
        description="Unique identifier of the order for which to retrieve status information."
    )


async def get_order_status_async(order_id: str) -> str:
    """
    Async retrieval of order status from a mock order tracking API.
    """
    api_key = ORDER_API_KEY
    if not api_key:
        raise ValueError("Order tracking API key is required")

    base_url = "https://api.example.com/orders/status"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    params = {"order_id": order_id}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                base_url,
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            return format_order_status_response(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Order '{order_id}' not found. Please check the order ID and try again."
        elif e.response.status_code == 401:
            return "Error: Unauthorized request. Please contact support."
        else:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"

    except httpx.RequestError as e:
        return f"Network error occurred while retrieving order status: {str(e)}"
    except Exception as e:
        return f"Unexpected error retrieving order status: {str(e)}"


def get_order_status(order_id: str) -> str:
    """
    Synchronous wrapper for the async order status function.
    """
    return asyncio.run(get_order_status_async(order_id))


def format_order_status_response(data: Dict[str, Any]) -> str:
    """Format the order status response into a readable message"""
    try:
        order_id = data["order_id"]
        status = data["status"]
        expected_delivery = data.get("expected_delivery", "N/A")
        return (
            f"Order {order_id} Status:\n"
            f"• Current Status: {status}\n"
            f"• Expected Delivery: {expected_delivery}"
        )
    except KeyError as e:
        return f"Error formatting order data: Missing field {str(e)}"


order_status_tool = StructuredTool.from_function(
    func=get_order_status,
    name="get_order_status",
    description="Retrieve current status and expected delivery information for a given order ID",
    args_schema=OrderStatusInput
)
