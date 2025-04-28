import copy
import json
import os
from typing import List

from pydantic import BaseModel, Field
from src.models import (
    CharacterSpecification,
    CharacterState,
    Character,
    EmotionalState,
    SocialRelationState,
    create_dynamic_enum
)

neutral_emotional_state = EmotionalState(
    joy="na",
    trust="na",
    fear="na",
    surprise="na",
    sadness="na",
    disgust="na",
    anger="na",
    anticipation="na",
)

class CharacterStore:
    """Read character specifications from given directory and initialize their state"""
    def __init__(self, character_dir: str, uids: List[str]):
        self.characters = {}
        
        # Iterate through uids
        for uid in uids:
            # Spec
            with open(os.path.join(character_dir, f"{uid}.json"), "r") as f:
                spec_dict = json.load(f)
            spec = CharacterSpecification(**spec_dict)
            
            # Emotion
            emotional_state = copy.deepcopy(neutral_emotional_state)
            
            # Social Relations
            social_relations = []
            for target_uid in uids:
                if uid==target_uid:
                    continue
                relation = SocialRelationState(
                    character_uid=target_uid,
                    emotion = copy.deepcopy(neutral_emotional_state),
                    knowledge = []
                )
                social_relations.append(relation)
            state = CharacterState(
                emotion=emotional_state,
                social_relations=social_relations
            )
            character = Character(
                uid=uid,
                spec=spec,
                state=state
            )
            self.characters[uid]=character
    
    def get(self, uid: str) -> Character:
        return self.characters[uid]
 