import json
import os
from enum import Enum
from typing import Any, Dict, List
import uuid
import yaml

from jinja2 import Template, StrictUndefined
from jinja2.exceptions import TemplateError

from openai import OpenAI, AsyncOpenAI
import pandas as pd
import pprint
from pydantic import BaseModel, Extra, Field
import tiktoken
from tqdm import tqdm

from config import settings

# 1. Load SimsConv dataset
# list of characters
character_dir = os.path.join(settings.data_dir, "story/SimsChat-60D0/characters")
character_fnames = [x for x in os.listdir(character_dir) if "txt" in x]
print(len(character_fnames), character_fnames[:5])


# 2. Load Prompt & Schema
## 2-1. Load Prompt
with open('prompts/structure_simschat_characters.yaml', 'r') as file:
    prompt = yaml.load(file, Loader=yaml.FullLoader)

system_message = prompt["system"]
user_template = Template(
    prompt['user'],
    undefined=StrictUndefined
)

def create_dynamic_enum(name: str, values: List[Any]) -> Enum:
    return Enum(name, {str(v): v for v in values})

## 2-2. Load Schema
# Predefined Aspects in Appendix C
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

defined_aspects = {
    "Gender": create_dynamic_enum("GenderAspect", gender),
    "Career": create_dynamic_enum("CareerAspect", career),
    "Aspiration": create_dynamic_enum("AspirationAspect", aspiration),
    "Trait": create_dynamic_enum("TraitAspect", trait),
    "Skill": create_dynamic_enum("SkillAspect", skill),
    "Emotion": create_dynamic_enum("EmotionAspect", emotion),
    "ConversationTopic": create_dynamic_enum("ConversationTopicAspect", conversation_topic),
}
class PersonalityTrait(BaseModel):
    trait: defined_aspects["Trait"]
    description: str
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
    
class CharacterSpecification(BaseModel):
    name: str
    gender: defined_aspects["Gender"]
    age: int
    dialogue_tone: str
    career: defined_aspects["Career"]
    personality_traits: List[PersonalityTrait]
    hobbies: List[defined_aspects["Skill"]]
    living_conditions: List[str]
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True


# 3. Generate
## Initialize OpenAI Client
openai_client = OpenAI(api_key=settings.openai_api_key)

def extract(character_description: str) -> CharacterSpecification:
    user_message = user_template.render(description=character_description)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    decode_params = {"temperature": 0.95}

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=CharacterSpecification,
        **decode_params,
    )
    return response.choices[0].message.parsed

character_collection = {
    "model": "gpt-4o",
    "source": {}
}

for character_fname in tqdm(character_fnames):
    character_uid = str(uuid.uuid4())
    with open(os.path.join(character_dir, character_fname), "r") as f:
        character_description = f.read()
        
    try:
        character = extract(character_description)
    except Exception as e:
        print(str(e))
        continue
    
    with open(f"assets/characters/{character_uid}.json", "w") as f:
        f.write(json.dumps(character.model_dump(), indent=4))
    
    character_collection["source"][character_uid] = character_fname
    
with open("assets/character_collection.json", "w") as f:
    f.write(json.dumps(character_collection, indent=4))