import itertools
from typing import Dict, List

from jinja2 import Template, StrictUndefined
from jinja2.exceptions import TemplateError
from openai import AsyncOpenAI
from pydantic import BaseModel, Extra, create_model

from src.character_store import CharacterStore
from src.models import (
    CharacterSpecification,
    CharacterState,
    Character,
    EmotionalState,
    SocialRelationState,
    create_dynamic_enum
)

        
class CharcterActionPredictor:
    def __init__(
        self,
        client: AsyncOpenAI,
        prompt: Dict[str, str],
        actions: Dict[str, str],
        character_store: CharacterStore,
        model: str="gpt-4.1-nano"
    ):
        self.client = client
        self.model = model
        
        # Load Prompt
        system_template = Template(
            prompt['system'],
            undefined=StrictUndefined
        )
        self.system_message = system_template.render(actions=actions)
        self.user_template = Template(
            prompt['user'],
            undefined=StrictUndefined
        )
        
        # Initialize Output Schema
        possible_actions_model = create_dynamic_enum(
            "PossibleCharacterActions",
            itertools.chain(*[
                [f"{k}-{a['action']}" for a in v]
                for k,v in actions.items()
            ])
        )
        
        self.character_action_model = create_model(
            "CharacterAction",
            __config__={"extra": "forbid"},
            **{
                "think": (str, ...),
                "action_type": (possible_actions_model, ...),
                "description": (str, ...),
                "targets": (List[str], ...),
            }
        )
        self.prediction_output_model = create_model(
            "ActionPredictionResult",
            __config__={"extra": "forbid"},
            **{
                "action": (self.character_action_model, ...),
                "updated_state": (CharacterState, ...)
            }
        )
        
        # character store
        self.character_store=character_store
    
    def _get_character(self, uid: str) -> Character:
        return self.character_store.get(uid)
    
    async def predict_action(
        self,
        scene_context: str,
        character: Character,
        # related_uids: List[str] = [],
    ):
        user_message = self.user_template.render(
            character_spec=character.spec.model_dump_json(),
            character_emotion=character.state.emotion.model_dump_json(),
            character_relations = [
                {
                    'character_name': self._get_character(x.character_uid).spec.name,
                    "relation": x.model_dump_json()
                }
                for x in character.state.social_relations
                # x.model_dump_json() for x in character.state.social_relations if x.character_uid in related_uids
            ],
            scene_context=scene_context
        )
        print(user_message)
        result = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message},
            ],
            response_format=self.prediction_output_model,
        )
        return result.choices[0].message.parsed
    
    def _update_character_state(self, character: Character, updated_state: CharacterState):
        character.state=updated_state
    
    async def forward(
        self,
        scene_context: str,
        uid: str,
        # related_uids: List[str] = [],
    ):
        character = self._get_character(uid)
        
        # Predict Character Action
        result = await self.predict_action(
            scene_context=scene_context,
            character=character,
            # related_uids=related_uids
        )
        
        # Update Character State
        self._update_character_state(
            character=character,
            updated_state=result.updated_state
        )
        return {
            "action": result.action.description,
            "action_type": result.action.action_type.value,
            "targets": result.action.targets
        }