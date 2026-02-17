import datetime
import random
import re
from typing import Dict, List

from pydantic import BaseModel, Field
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError


SERVER_DESCRIPTION = """Provides weather related tools"""

GET_CURRENT_WEATHER_TOOL_DESCRIPTION = (
    "Get current weather for a city. "
    "Returns WeatherResult with time format YYYYMMDD-HH:00."
)

GET_WEATHER_FORECAST_TOOL_DESCRIPTION = (
    "Get multi-day forecast for a city. "
    "Returns List[WeatherResult] with time format YYYYMMDD."
)

mcp = FastMCP(
    "weather-service",
    instructions=SERVER_DESCRIPTION,
)

########################################
# Data Models
########################################

class WeatherStatus(BaseModel):
    temp_cel: int = Field(..., description="Temperature in celsius")
    condition: str = Field(..., description="Weather condition")
    humidity_pct: int = Field(..., description="Humidity Percent (0~100)")
    wind_kph: int = Field(..., description="Wind speed in kph")


class WeatherResult(BaseModel):
    time: str = Field(..., description="Time of prediction")
    status: WeatherStatus = Field(..., description="Weather status")


########################################
# Dummy Data
########################################

_WEATHER_DB: Dict[str, WeatherStatus] = {
    "Seoul": WeatherStatus(temp_cel=2, humidity_pct=55, condition="맑음", wind_kph=12),
    "Busan": WeatherStatus(temp_cel=7, humidity_pct=60, condition="구름 조금", wind_kph=18),
    "Incheon": WeatherStatus(temp_cel=1, humidity_pct=58, condition="흐림", wind_kph=20),
    "Suwon": WeatherStatus(temp_cel=1, humidity_pct=52, condition="맑음", wind_kph=8),
    "Seongnam": WeatherStatus(temp_cel=2, humidity_pct=54, condition="구름 많음", wind_kph=9),
    "Ulsan": WeatherStatus(temp_cel=6, humidity_pct=62, condition="흐림", wind_kph=15),
    "Gimhae": WeatherStatus(temp_cel=7, humidity_pct=58, condition="맑음", wind_kph=11),
    "Changwon": WeatherStatus(temp_cel=6, humidity_pct=57, condition="구름 조금", wind_kph=13),
}

_CONDITIONS_CYCLE = ["맑음", "구름 조금", "구름 많음", "흐림", "비"]

_KO_TO_EN_CITY = {
    "서울": "Seoul",
    "부산": "Busan",
    "인천": "Incheon",
    "수원": "Suwon",
    "성남": "Seongnam",
    "울산": "Ulsan",
    "김해": "Gimhae",
    "창원": "Changwon",
}


########################################
# Helpers
########################################

def alphabet_ratio(x: str) -> float:
    characters = [c for c in x if c.isalpha()]
    x = "".join(characters)
    if not x:
        return 0.0
    letters = re.findall(r"[A-Za-z]", x)
    return len(letters) / len(x)


def is_name_english(city_name: str) -> bool:
    return alphabet_ratio(city_name) >= 0.8


def resolve_city_name(city_name: str) -> str:
    if not city_name or not city_name.strip():
        raise ToolError("city_name is required")

    raw = city_name.strip()

    if raw in _KO_TO_EN_CITY:
        return _KO_TO_EN_CITY[raw]

    if is_name_english(raw):
        lower = raw.lower()
        for k in _WEATHER_DB:
            if k.lower() == lower:
                return k

    raise ToolError(f"Unknown city: {city_name}")


########################################
# Business Logic
########################################

def get_current(city_name: str) -> WeatherResult:
    city_key = resolve_city_name(city_name)
    base = _WEATHER_DB[city_key]

    temp = base.temp_cel + random.randint(-1, 1)
    humidity = base.humidity_pct + random.randint(-3, 3)
    wind = base.wind_kph + random.randint(-2, 2)

    condition = base.condition

    now = datetime.datetime.now()

    # YYYYMMDD-HH:00
    time_str = now.strftime("%Y%m%d-%H:00")

    return WeatherResult(
        time=time_str,
        status=WeatherStatus(
            temp_cel=temp,
            condition=condition,
            humidity_pct=max(0, min(100, humidity)),
            wind_kph=max(0, wind),
        ),
    )


def get_forecast(city_name: str, days: int = 3) -> List[WeatherResult]:
    if days < 1 or days > 14:
        raise ToolError("days must be between 1 and 14")

    city_key = resolve_city_name(city_name)
    base = _WEATHER_DB[city_key]

    today = datetime.date.today()
    results: List[WeatherResult] = []

    city_hash = abs(hash(city_key)) % 10000

    for d in range(days):
        date = today + datetime.timedelta(days=d)

        temp = base.temp_cel + ((d + city_hash) % 5) - 2
        condition = _CONDITIONS_CYCLE[(d + city_hash) % len(_CONDITIONS_CYCLE)]
        humidity = base.humidity_pct + ((d * 7 + city_hash) % 11) - 5
        wind = base.wind_kph + ((d * 5 + city_hash) % 9) - 4

        # YYYYMMDD
        time_str = date.strftime("%Y%m%d")

        results.append(
            WeatherResult(
                time=time_str,
                status=WeatherStatus(
                    temp_cel=temp,
                    condition=condition,
                    humidity_pct=max(0, min(100, humidity)),
                    wind_kph=max(0, wind),
                ),
            )
        )

    return results


########################################
# MCP Tools
########################################

@mcp.tool(
    name="get_current_weather",
    description=GET_CURRENT_WEATHER_TOOL_DESCRIPTION,
    meta={"version": "1.0", "author": "id4thomas"},
)
def tool_get_current_weather(city_name: str) -> WeatherResult:
    return get_current(city_name)


@mcp.tool(
    name="get_weather_forecast",
    description=GET_WEATHER_FORECAST_TOOL_DESCRIPTION,
    meta={"version": "1.0", "author": "id4thomas"},
)
def tool_get_weather_forecast(city_name: str, days: int = 3) -> List[WeatherResult]:
    return get_forecast(city_name, days)


########################################
# ASGI App
########################################

app = mcp.http_app()
