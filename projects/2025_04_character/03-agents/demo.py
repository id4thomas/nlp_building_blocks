import json
import random
from typing import Any, List
import yaml

import gradio as gr
from jinja2 import Template, StrictUndefined
from openai import AsyncOpenAI
import pandas as pd
from pydantic import BaseModel, Extra, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from src.models import (
    CharacterSpecification,
    CharacterState,
    Character,
    EmotionalState,
    SceneBackground,
    create_dynamic_enum
)
from src.action_predictor import (
    CharcterActionPredictor
)
from src.character_store import CharacterStore

from config import settings

# 1. Initalize LLM
print(settings.llm_model)

# Pydantic Model
provider = OpenAIProvider(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key
)
model = OpenAIModel(
    model_name=settings.llm_model,
    provider=provider
)

# OpenAI Client
client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key
)

# 2. Load Information
## Load Scene Info
scene_outline="Inside the bustling grandeur of the metropolitan fashion gala, Zephyr Orion, a 28-year-old jocular astronaut with a penchant for playful storytelling, encounters Vivienne LaRoux. Vivienne, also 28, exudes sophistication and an assertive demeanor as a renowned style influencer. Initially, she greets Zephyr's lighthearted banter with icy indifference, her mean streak surfacing sporadically. However, Zephyr's infectious humor gradually softens her edges, revealing a subtly receptive side. Their verbal dance, rich with lively exchanges, challenges both to reconsider their outlooks, Zephyr embracing Vivienne's world of high fashion while she discovers a brighter perspective in his social magnetism."
scene_background = {
    "location": "The Metropolitan Fashion Gala",
    "setting": "An opulent event hall adorned with sparkling chandeliers and cutting-edge fashion displays.",
    "explanation": "In the midst of a vibrant and luxurious fashion gala, Zephyr Orion and Vivienne LaRoux engage in a compelling social interaction. Zephyr, with his distinctive brand of humor and affable nature, engages Vivienne in a conversation that is both disarming and endearing. Their exchange, a medley of banter and keen observations, reflects not only their mutual intrigue but also hints at a budding dynamic that bridges their contrasting worlds of space and style. As the evening unfolds, both characters experience subtle shifts in their perspectives, opening their minds to each other's unique lives, and suggesting the start of a meaningful connection."
}
scene_background = SceneBackground(**scene_background)

## Load Character Info
character_uids = [
    "35f0c56f-263d-42df-846c-e1833d8ca0ab",
    "00d66087-9b3b-46da-bd74-bf45cbe81d3c"
]
character_uid_enum = create_dynamic_enum("CharacterUID", character_uids)

character_store = CharacterStore(
    character_dir="assets/characters",
    uids=character_uids
)
# get scene characters
scene_characters = {
    uid: character_store.get(uid).spec
    for uid in character_uids
}

## Load Actions
with open("assets/sims_interactions.json", "r") as f:
    all_actions = json.load(f)
n = 10
actions = {k: random.sample(v, min(len(v), n)) for k,v in all_actions.items()}
print(actions.keys())

# 3. Initialize Agent
ACTION_HISTORY = []

## 3-1. Initialize Action Predictor
with open('prompts/action_predictor.yaml', 'r') as file:
    action_predictor_prompt = yaml.load(file, Loader=yaml.FullLoader)

predictor = CharcterActionPredictor(
    client=client,
    prompt=action_predictor_prompt,
    actions=actions,
    character_store=character_store,
    model=settings.llm_model
)
class Action(BaseModel):
    character_uid: str
    action: str
    action_type: str
    targets: List[str]
    
    class Config:
        extra = "forbid"
        use_enum_values = True
    
with open('prompts/planning_agent.yaml', 'r') as file:
    agent_prompt = yaml.load(file, Loader=yaml.FullLoader)

user_template  = Template(
    agent_prompt['user'],
    undefined=StrictUndefined
)

agent = Agent(
    model=model,
    name="planning",
    output_type=Action,
    system_prompt = agent_prompt['system']
)

## 3-2. register tool
@agent.tool
async def predict_action(
    ctx: RunContext[None],
    scene_context: str,
    character_uid: character_uid_enum,
    # related_character_uids: List[str]
) -> Action:
    
    predicted_action = await predictor.forward(
        scene_context=scene_context,
        uid=character_uid.value,
        # related_uids=related_character_uids
    )
    action = Action(
        character_uid=character_uid.value,
        action=predicted_action['action'],
        action_type=predicted_action['action_type'],
        targets=predicted_action['targets']
    )
    return action

def parse_action(action: Action):
    return {
        "character": character_store.get(action.character_uid).spec.name,
        "action": action.action,
        "action_type": action.action_type,
        "targets": ",".join(action.targets)
        # "targets": ",".join([
        #     character_store.get(x).spec.name
        #     for x in action.targets
        # ])
    }
    

def log_action():
    history_dict = {
        "step": [],
        "character": [],
        "targets": [],
        "action": [],
        "type": [],
    }
    for action_i, action in enumerate(ACTION_HISTORY):
        parsed_action = parse_action(action)
        
        history_dict["step"].append(action_i+1)
        history_dict["character"].append(parsed_action['character'])
        history_dict["targets"].append(parsed_action["targets"])
        history_dict["action"].append(parsed_action["action"])
        history_dict["type"].append(parsed_action["action_type"])
        
    df = pd.DataFrame.from_dict(history_dict)
    def highlight(val):
        if 'Zephyr' in val:
            return 'background-color: lightgreen'
        elif 'Vivienne' in val:
            return 'background-color: yellow'
        else:
            return ''  # no formatting for other values

    # apply to column 'x' only
    df = df.style.applymap(highlight, subset=['character'])

    return df
    

async def plan():
    contents = {
        "scene_outline": scene_outline,
        "scene_background": scene_background,
        "characters": scene_characters,
        "history": ACTION_HISTORY
    }
    user_message = user_template.render(**contents)
    result = await agent.run(user_message)
    predicted_action = result.output
    ACTION_HISTORY.append(predicted_action)
    df = log_action()
    return df


def main():
    # with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:
    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("# Story Character Agent")
        
        with gr.Row():    
            with gr.Column(scale=3):
                # Scene Info
                gr.Markdown("## Scene Information")
                gr.JSON(
                    json.dumps(
                        {
                            "outline": scene_outline,
                            "background": scene_background.model_dump()
                        },
                        indent=2,
                    )
                )
                gr.Markdown("## Character Information")
                gr.JSON(
                    json.dumps(
                        {k: v.model_dump() for k,v in scene_characters.items()},
                        indent=2,
                    )
                )
            
            with gr.Column(scale=7):
                gr.Markdown("## Action Prediction")
            
                retrieve_button = gr.Button("Predict", variant="primary")
                history = gr.Dataframe(
                    wrap=True,
                    label="Action History)",
                    max_height=1000,
                    column_widths=["6%", "16%", "16%", "47%", "15%"],
                    datatype=["int", "str", "str", "str", "str"]
                )
                retrieve_button.click(
                    plan, 
                    inputs=[], 
                    outputs=[history], 
                    api_name=False
                )
                
            
        
        demo.launch()

if __name__=="__main__":
    main()