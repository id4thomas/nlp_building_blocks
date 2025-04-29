from typing import List
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

class Action(BaseModel):
    character_uid: str
    action: str
    targets: List[str]
    
    class Config:
        extra = "forbid"
        use_enum_values = True
        
class StoryPlanner:
    def __init__(
        self,
        action_predictor
    ):
        self.action_predictor = action_predictor

    async def predict_action(
        ctx: RunContext[None],
        scene_context: str,
        character_uid: str,
        related_character_uids: List[str]
    ) -> Action:
        
        predicted_action = await predictor.forward(
            scene_context=scene_context,
            uid="35f0c56f-263d-42df-846c-e1833d8ca0ab",
            related_uids=related_character_uids
        )
        action = Action(
            character_uid=character_uid,
            action=predicted_action['action'],
            targets=predicted_action['targets']
        )
        return action