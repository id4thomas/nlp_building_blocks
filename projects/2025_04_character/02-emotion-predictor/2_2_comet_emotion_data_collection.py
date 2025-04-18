'''Data Collection - gpt-4.1-nano-2025-04-14
# Train
100%|███| 377/377 [05:22<00:00,  1.17it/s]
Processed 377 files.
- 1.4M tokens ($0.91)

# Dev
| 42/42 [01:13<00:00,  1.74s/it]

# Test
| 47/47 [01:03<00:00,  1.36s/it]
'''

import asyncio
from enum import Enum
import json
import os
from typing import Any, List

from openai import AsyncOpenAI
import pandas as pd

from pydantic import BaseModel, Extra, Field
from pydantic import create_model
from tqdm.asyncio import tqdm_asyncio

from config import settings

print(settings.llm_model)
client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key
)

class RelationshipStatus(str, Enum):
    na = "na"
    low = "low"
    medium = "medium"
    high = "high"
    
class EmotionLabel(BaseModel):
    joy: RelationshipStatus
    trust: RelationshipStatus
    fear: RelationshipStatus
    surprise: RelationshipStatus
    sadness: RelationshipStatus
    disgust: RelationshipStatus
    anger: RelationshipStatus
    anticipation: RelationshipStatus
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
        
class EntryResult(BaseModel):
    # uid: create_dynamic_enum("uid", uids)
    source: str
    character: str
    emotion: EmotionLabel
    reason: str
    
    class Config:
        extra = Extra.forbid
        use_enum_values = True
        
# Multiple Samples
SYSTEM_PROMPT = '''You are provided with entries from a commonsense reasoning dataset (COMET-2020) that consist of three parts:
1.	source: A short event description containing the placeholder “PersonX” or “PersonY.”
2.	relation: Either xReact (indicating the subject’s emotional reaction) or oReact (indicating the other person’s emotional reaction).
3.	target: A brief description of the emotion, state, or reaction.

# Task
For each entry in the dataset, you need to:
## Expand the Source
Replace “PersonX” or “PersonY” with a random human name (e.g., Alex, Jordan, María, Fatima, Ethan, etc.) and expand the event into a realistic scenario by adding enough context.
Write at least 3 sentences with detailed descriptions of the situation. (about 300 characters)
For example, if the source is “PersonX abandons ___ altogether,” you might expand it to:
“Veronica decides to stop relying on social media altogether, feeling that she has complete control over her personal life. ...”

## Map the Emotion: Map the target emotion onto the 8 primary emotions from Plutchik’s Wheel
* Joy
* Trust
* Fear
* Surprise
* Sadness
* Disgust
* Anger
* Anticipation

For each emotion, assign one of the following intensities:
* "na" (not applicable)
* "low"
* "medium"
* "high"

Interpret the target emotion (e.g., “authoritative”) in terms of Plutchik’s emotions. For example, you might decide:
* trust: high
* joy: low
* anticipation: medium

## Write a Reason:
Provide a one-sentence rationale ("reason") explaining why the subject (if xReact) or the other person (if oReact) feels the given emotion(s).
ex. “She feels empowered and confident after cutting out social media.”

# Note
* For xReact, the emotion data should pertain to the subject’s (formerly “PersonX”) reaction.
* For oReact, the emotion data should pertain to the other person (formerly “PersonY”)

Return in the following JSON format
* write prediction for each entry by its uid as key
{
    {uid_value}: {
      "source": "Expanded realistic scenario with a real name in place of PersonX/PersonY",
      "person": "name of the person that will feel this emotion",
      "emotion": {
        "joy": "na" | "low" | "medium" | "high",
        "trust": "na" | "low" | "medium" | "high",
        "fear": "na" | "low" | "medium" | "high",
        "surprise": "na" | "low" | "medium" | "high",
        "sadness": "na" | "low" | "medium" | "high",
        "disgust": "na" | "low" | "medium" | "high",
        "anger": "na" | "low" | "medium" | "high",
        "anticipation": "na" | "low" | "medium" | "high"
      },
      "reason": "One sentence explaining why these emotions occur"
    },
    ...
}

Only return the JSON, don't return in markdown format. (ex. "```json...") start like "{..."
'''

USER_TEMPLATE = '''[Entries]
{entries}'''


def prepare_request(data):
    uids: List[str] = [v for k,v in data['uids'].items()]
    entries = data['entries']
    
    # Prepare response format
    spec_dict = dict()
    for uid in uids:
        spec_dict[uid] = (EntryResult, ...)

    Results = create_model("Results", __config__={"extra": "forbid"}, **spec_dict)
    user_message = USER_TEMPLATE.format(
        entries = json.dumps(entries)
    )
    return user_message, Results


async def predict(data):
    user_message, Results = prepare_request(data)
    completion_result = await client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=Results,
    )
    prediction_result = completion_result.choices[0].message.parsed
    return prediction_result

async def predict_with_semaphore(sem, fname, data):
    async with sem:
        return fname, await predict(data)

async def main():
    split = "test" # "train", "dev", "test"
    request_dir = f"data/comet/{split}/request"
    output_dir = f"data/comet/{split}/response/{settings.llm_model}"
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(32)  
    
    ## 1. Prepare Input Batches
    fnames = [x for x in os.listdir(request_dir) if ".json" in x]#[:2]
    tasks = []
    for fname in fnames:
        with open(os.path.join(request_dir, fname), "r") as f:
            data = json.load(f)
        
        tasks.append(
            asyncio.ensure_future(
                predict_with_semaphore(sem, fname, data)
            )
        )
    results = await tqdm_asyncio.gather(*tasks)
    for fname, result in results:
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(result.model_dump(), f, indent=2)
    print(f"Processed {len(results)} files.")

if __name__ ==  '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())