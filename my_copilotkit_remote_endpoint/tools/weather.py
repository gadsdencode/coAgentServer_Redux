# tools/weather.py
from typing import Any, Dict
from langchain.tools import StructuredTool
import httpx
from pydantic import BaseModel, Field
import os
import asyncio
from functools import partial

API_KEY = os.getenv('OPENWEATHER_API_KEY')


class WeatherInput(BaseModel):
    """Schema for weather request parameters"""
    query: str = Field(
        ...,
        description="Location query (city name, zip code, or coordinates)"
    )


async def get_weather_async(query: str) -> str:
    """
    Async implementation of weather retrieval.
    """
    api_key = API_KEY
    if not api_key:
        raise ValueError("OpenWeatherMap API key is required")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    headers = {"Content-Type": "application/json"}

    try:
        params = {
            "q": query,
            "appid": api_key,
            "units": "metric"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                base_url,
                params=params,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            return format_weather_response(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Location '{query}' not found. Please check the location and try again."
        elif e.response.status_code == 401:
            return "Error: Invalid API key. Please contact support."
        else:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"

    except httpx.RequestError as e:
        return f"Network error occurred: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_weather(query: str) -> str:
    """
    Synchronous wrapper for the async weather function.
    """
    return asyncio.run(get_weather_async(query))


def format_weather_response(data: Dict[str, Any]) -> str:
    """Format the weather API response into a human-readable string"""
    try:
        location = data["name"]
        country = data["sys"]["country"]
        temp = round(data["main"]["temp"], 1)
        feels_like = round(data["main"]["feels_like"], 1)
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"Current weather in {location}, {country}:\n"
            f"• Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"• Conditions: {description.capitalize()}\n"
            f"• Humidity: {humidity}%"
        )
    except KeyError as e:
        return f"Error formatting weather data: Missing field {str(e)}"


# Create the tool using StructuredTool with the sync wrapper
weather_tool = StructuredTool.from_function(
    func=get_weather,  # Use the sync wrapper
    name="get_weather",
    description="Retrieve current weather information based on location name, postal code, coordinates, or IP address",
    args_schema=WeatherInput
)
