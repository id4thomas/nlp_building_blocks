from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from .constants import (
    GENDER,
    CAREER,
    ASPIRATION,
    TRAITS,
    SKILLS,
    CONVERSATION_TOPICS,
    emotion_descriptions
)

def create_dynamic_enum(name: str, values: List[Any]) -> Enum:
    return Enum(name, {str(v): v for v in values})

DEFINED_ASPECTS = {
    "Gender": create_dynamic_enum("GenderAspect", GENDER),
    "Career": create_dynamic_enum("CareerAspect", CAREER),
    "Aspiration": create_dynamic_enum("AspirationAspect", ASPIRATION),
    "Trait": create_dynamic_enum("TraitAspect", TRAITS),
    "Skill": create_dynamic_enum("SkillAspect", SKILLS),
    # "Emotion": create_dynamic_enum("EmotionAspect", emotion),
    "ConversationTopic": create_dynamic_enum("ConversationTopicAspect", CONVERSATION_TOPICS),
}

class Polarity(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"

class PersonalityTrait(BaseModel):
    trait: DEFINED_ASPECTS["Trait"]
    description: str
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
    
class SocialRelationship(BaseModel):
    target: str
    status: Polarity
    description: str
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True

class CharacterSpecification(BaseModel):
    name: str
    gender: DEFINED_ASPECTS["Gender"]
    age: int
    dialogue_tone: str
    career: DEFINED_ASPECTS["Career"]
    personality_traits: List[PersonalityTrait]
    hobbies: List[DEFINED_ASPECTS["Skill"]]
    living_conditions: List[str]
    # social_relationships: List[SocialRelationship]
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
        

class EmotionPolarity(str, Enum):
    na = "na"
    low = "low"
    medium = "medium"
    high = "high"
    
class EmotionalState(BaseModel):
    joy: EmotionPolarity
    trust: EmotionPolarity
    fear: EmotionPolarity
    surprise: EmotionPolarity
    sadness: EmotionPolarity
    disgust: EmotionPolarity
    anger: EmotionPolarity
    anticipation: EmotionPolarity


class SocialRelationState(BaseModel):
    """1-directional Relationship state between 2 characters"""
    character_uid: str
    emotion: EmotionalState
    knowledge: List[str]
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True

class CharacterState(BaseModel):
    # sentiment: Polarity
    emotion: EmotionalState
    social_relations: List[SocialRelationState]
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True

class Character(BaseModel):
    uid: str
    spec: CharacterSpecification
    state: CharacterState
    
    
# Scene
class SceneBackground(BaseModel):
    location: str
    setting: str
    explanation: str