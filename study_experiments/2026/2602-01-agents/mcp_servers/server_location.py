import asyncio
import math
import random
import re

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

from pydantic import BaseModel, Field


#### Description Texts
SERVER_DESCRIPTION='''Provides location related tools'''

GET_USER_LOCATION_TOOL_DESCRIPTION = "Receives user's id info and returns their current location in city name, coordinates"
GET_NEARBY_CITIES_TOOL_DESCRIPTION = "Receives city info and returns nearby n cities"

#### Initialize Server
mcp = FastMCP(
    "location-service",
    instructions=SERVER_DESCRIPTION
)

#### Data Models
class Coordinate(BaseModel):
    lat: float = Field(..., description="Coordinate latitude value")
    lon: float = Field(..., description="Coordinate longitude value")
    

class City(BaseModel):
    name: str = Field(..., description="City name in english")
    coordinate: Coordinate = Field(..., description="City coordinate value")
    
#### Logic Functions
ALLOWED_USERS = {"1234"}

CITY_DB = {
    "Seoul": City(
        name="Seoul",
        coordinate=Coordinate(lat=37.5665, lon=126.9780)
    ),
    "Busan": City(
        name="Busan",
        coordinate=Coordinate(lat=35.1796, lon=129.0756)
    ),
    "Suwon": City(
        name="Suwon",
        coordinate=Coordinate(lat=37.2636, lon=127.0286)
    ),
    "Seongnam": City(
        name="Seongnam",
        coordinate=Coordinate(lat=37.4200, lon=127.1267)
    ),
    "Ulsan": City(
        name="Ulsan",
        coordinate=Coordinate(lat=35.5384, lon=129.3114)
    ),
    "Gimhae": City(
        name="Gimhae",
        coordinate=Coordinate(lat=35.2285, lon=128.8894)
    ),
    "Changwon": City(
        name="Changwon",
        coordinate=Coordinate(lat=35.2280, lon=128.6811)
    ),
}

def validate_user(user_id: str) -> bool:
    return user_id in ALLOWED_USERS

def alphabet_ratio(x: str) -> float:
    characters = [c for c in x if c.isalpha()]
    x = ''.join(characters)
    if not x:
        return 0.0
    
    letters = re.findall(r"[A-Za-z]", x)
    return len(letters) / len(x)

def is_name_english(city_name: str) -> bool:
    # Check if name is in english
    if alphabet_ratio(city_name)<0.8:
        return False
    else:
        return True

def get_user_location(user_id: str) -> City:
    """Dummy Function"""
    candidates = [
        CITY_DB["Seoul"],
        CITY_DB["Busan"],
    ]
    return random.sample(candidates, k=1)[0]

def haversine(c1: Coordinate, c2: Coordinate) -> float:
    """Distance in meters between two WGS84 coordinates."""
    radius_m = 6_371_000

    phi1 = math.radians(c1.lat)
    phi2 = math.radians(c2.lat)
    d_phi = math.radians(c2.lat - c1.lat)
    d_lng = math.radians(c2.lon - c1.lon)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lng / 2) ** 2
    )

    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_nearby_cities(
    city_name: str,
    n: int = 3,
    include_self: bool = False,
) -> list[tuple[str, City, float]]:
    """
    Return n nearest cities to `city_key`.

    Returns list of (key, City, distance_m) sorted by distance ascending.
    """
    if city_name not in CITY_DB:
        raise KeyError(f"Unknown city key: {city_name}")

    src = CITY_DB[city_name]

    results = []

    for key, city in CITY_DB.items():
        if not include_self and key == city_name:
            continue

        dist_m = haversine(src.coordinate, city.coordinate)
        results.append((key, city, dist_m))

    results.sort(key=lambda x: x[2])
    return results[:max(0, n)]

#### MCP Tool Definition

@mcp.tool(
    name="get_user_location",
    description=GET_USER_LOCATION_TOOL_DESCRIPTION,
    tags={"user"},
    meta={"version": "1.0", "author": "id4thomas"}
)
def tool_get_user_location() -> City:
    ## WARNING: Proper Authentication implementation is required
    headers = get_http_headers()  # 없으면 {}
    user_id = headers.get("x-user-id")

    # 1. Validate User
    is_valid_user = validate_user(user_id)
    if not is_valid_user:
        raise ToolError(f"User {user_id} is not a valid user")

    # 2. Get User Location (Dummy)
    location = get_user_location(user_id)
    return location

@mcp.tool(
    name="get_nearby_cities",
    description=GET_NEARBY_CITIES_TOOL_DESCRIPTION,
    tags={},
    meta={"version": "1.0", "author": "id4thomas"}
)
def tool_get_nearby_cities(city_name: str, n: int = 3) -> list[City]:
    if not is_name_english(city_name):
        raise ToolError("City name must be provided in english.")

    if city_name not in CITY_DB:
        raise ToolError(f"City {city_name} not recognized.")
    
    results = get_nearby_cities(
        city_name=city_name,
        n=n
    )
    cities = [x[1] for x in results]
    return cities


# If running standalone without uvicorn
# async def main():
#     '''
#     mcp.run documentation 잘못됨 주의
#     - https://github.com/jlowin/fastmcp/issues/873
#     '''
#     # mcp.run(transport="streamable-http")
#     # fastmcp >= 3.0 기준
#     await mcp.run_http_async(
#         transport="streamable-http",
#         host="0.0.0.0",
#         port=8001
#     )


# if __name__ == "__main__":
#     asyncio.run(main())

# For use with uvicorn
app = mcp.http_app()