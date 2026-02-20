from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

from builder import AgentBuilder

agent = AgentBuilder.build(
    name="WeatherAgent",
    user_id="1234",
    tool_names=["location", "weather"],
    user_name="id4thomas"
)


app = FastAPI(title="Weather Agent (Microsoft Agent Framework)")
add_agent_framework_fastapi_endpoint(app=app, agent=agent, path="/")
