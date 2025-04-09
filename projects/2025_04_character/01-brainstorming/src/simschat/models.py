from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, Field

def create_dynamic_enum(name: str, values: List[Any]) -> Enum:
    return Enum(name, {str(v): v for v in values})

# Character Information
## Predefined Aspects in Appendix C
gender = ["male", "female"]

career = [
    "Actor", "Astronaut", "Athlete", "Business", "Civil Designer", "Conservationist",
    "Criminal", "Critic", "Culinary", "Detective", "Doctor", "Education", "Engineer",
    "Entertainer", "Freelancer", "Gardener", "Law", "Military", "Painter", "Politician",
    "Scientist", "Social Media", "Secret Agent", "Style Influencer", "Tech Guru", "Writer"
]

aspiration = [
    "Athletic", "Cheerful", "Deviance", "Family", "Food", "Fortune", "Knowledge",
    "Love", "Nature", "Popularity"
]

trait = [
    "Ambitious", "Cheerful", "Childish", "Clumsy", "Creative", "Erratic", "Genius",
    "Gloomy", "Goofball", "Hot-Headed", "Romantic", "Self-Assured", "Bro", "Evil",
    "Family-Oriented", "Good", "Hates Children", "Jealous", "Loner", "Loyal", "Mean",
    "Noncommittal", "Outgoing", "Snob", "Active", "Glutton", "Kleptomaniac", "Lazy",
    "Materialistic", "Neat", "Perfectionist", "Slob", "Vegetarian", "Art Lover",
    "Bookworm", "Foodie", "Geek", "Loves the Outdoors", "Music Lover"
]

skill = [
    "Acting", "Archaeology", "Baking", "Bowling", "Charisma", "Comedy", "Cooking",
    "Cross-Stitch", "DJ Mixing", "Dancing", "Fabrication", "Fishing", "Fitness",
    "Flower Arranging", "Gardening", "Gourmet Cooking", "Guitar", "Handiness",
    "Herbalism", "Juice Fizzing", "Logic", "Media Production", "Mischief", "Mixology",
    "Painting", "Parenting", "Pet Training", "Photography", "Piano", "Pipe Organ",
    "Programming", "Rock Climbing", "Rocket Science", "Selvadoradian Culture", "Singing",
    "Vampiric Lore", "Veterinarian", "Video Gaming", "Violin",
    "Wellness", "Writing"
]

emotion = [
    "Angry", "Asleep", "Bored", "Confident", "Dazed", "Embarrassed",
    "Energized", "Fine", "Flirty", "Focused", "Happy", "Inspired",
    "Playful", "Sad", "Tense", "Uncomfortable"
]

conversation_topic = [
    "affection", "arguments", "complaints", "compliments", "deception", "deep thoughts",
    "discussing hobbies", "discussing interests", "flirtation", "gossip", "jokes",
    "malicious interactions", "physical intimacy", "potty humor", "pranks", "silly behavior",
    "small talk", "stories"
]

DEFINED_ASPECTS = {
    "Gender": create_dynamic_enum("GenderAspect", gender),
    "Career": create_dynamic_enum("CareerAspect", career),
    "Aspiration": create_dynamic_enum("AspirationAspect", aspiration),
    "Trait": create_dynamic_enum("TraitAspect", trait),
    "Skill": create_dynamic_enum("SkillAspect", skill),
    "Emotion": create_dynamic_enum("EmotionAspect", emotion),
    "ConversationTopic": create_dynamic_enum("ConversationTopicAspect", conversation_topic),
}

class PersonalityTrait(BaseModel):
    trait: DEFINED_ASPECTS["Trait"]
    description: str
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True

class RelationshipStatus(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    
class SocialRelationship(BaseModel):
    target: str
    status: RelationshipStatus
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
    social_relationships: List[SocialRelationship]
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
        
# State Information
## Scene State
# class SceneState(BaseModel):
#     location: str
#     setting: str
#     explanation: str
    
#     class Config:
#         extra = Extra.forbid
#         use_enum_values = True

## Character State
### plutchik's wheel of emotion to represent emotional state
# emotional_state = [
#     "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"
# ]
# EmotionalState = create_dynamic_enum("EmtionalState", emotional_state)

# class CharacterRelationState(BaseModel):
#     """Relationship state between 2 characters"""
#     character: str
#     emotion: EmotionalState
#     knowledge: List[str]
    
#     class Config:
#         extra = Extra.forbid
#         use_enum_values = True

# class CharacterState(BaseModel):
#     emotion: EmotionalState
#     social_relations: List[CharacterRelationState]
    
#     class Config:
#         extra = Extra.forbid
#         use_enum_values = True
    