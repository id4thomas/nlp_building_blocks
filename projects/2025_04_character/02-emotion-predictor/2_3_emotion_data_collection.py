'''Data Collection -
* Total about 1.764M input tokens (14,862 entries)
[gpt-4.1-mini-2025-04-14] ($3.63)
# Train (12,061)
| 377/377 [13:36<00:00,  2.17s/it]

# Dev (1,317)
| 42/42 [01:59<00:00,  2.85s/it]

# Test (1,484)
| 47/47 [02:11<00:00,  2.80s/it]

[gpt-4.1-nano-2025-04-14] ($1.12)
# Train
| 377/377 [05:22<00:00,  1.17it/s]
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

## Map the Emotion:
Map the target emotion onto the 8 primary emotions from Plutchik’s Wheel
* Interpret the entry's 'target' reaction in terms of Plutchik’s 8 primary emotions.

[8 Primary Emotions]
* "joy": Joy is a bright, uplifting emotion that reflects happiness, satisfaction, and a sense of well-being. It often arises when our desires are fulfilled or we experience positive moments, and it helps energize both our minds and bodies. Joy can enhance social connections and overall resilience by radiating positivity
* "trust": Trust is the reassuring feeling of confidence and security in another person or situation. It builds from consistent, reliable interactions and underpins strong, supportive relationships. This emotion fosters cooperation and reduces anxiety by creating a sense of safety
* "fear": Fear is an instinctive response to perceived threats that activates our fight-or-flight mechanism. It heightens awareness and prepares our body to respond quickly to danger, making it essential for survival. Despite its discomfort, fear is a crucial signal that prompts protective action and risk assessment
* "surprise": Surprise occurs when we encounter the unexpected, momentarily halting our regular thought process. This emotion can be positive, neutral, or even negative, depending on the context, and often sparks curiosity about what comes next. Its brief nature helps redirect our focus and encourages adaptive responses to new situations
* "sadness": Sadness is a deep, reflective emotion that often emerges from loss, disappointment, or unmet expectations. It can lead to introspection and a desire for support as we navigate feelings of grief or dejection. Although challenging, sadness can also foster empathy and pave the way for emotional healing and growth
* "disgust": Disgust is an aversive emotion that signals rejection toward something perceived as harmful, unclean, or morally offensive. It serves as a protective mechanism, prompting us to avoid substances or situations that might be dangerous. This emotion plays a vital role in maintaining both physical health and ethical boundaries
* "anger": Anger arises when we perceive injustice, frustration, or a threat to our well-being, often urging us to act in response. It can manifest as physical tension and heightened energy, signaling that something in our environment needs to change. When managed effectively, anger can motivate constructive action and help assert personal boundaries
* "anticipation": Anticipation is the forward-looking emotion characterized by a mix of excitement and apprehension about future events. It motivates preparation and planning while balancing hope with cautious vigilance. This emotion bridges the gap between our present state and the potential for positive outcomes in the future

For each emotion, assign one of the following intensities:
* "na" (not applicable)
* "low"
* "medium"
* "high"

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

async def predict_with_semaphore(sem, fname, data, output_dir):
    async with sem:
        result = await predict(data)
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        return fname, result

async def main():
    split = "test" # "train", "dev", "test"
    request_dir = f"data/comet/{split}/request"
    output_dir = f"data/comet/{split}/response/{settings.llm_model}"
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(32)  
    
    ## 1. Prepare Input Batches
    
    fnames = [x for x in os.listdir(request_dir) if ".json" in x]#[:2]
    fnames = [x for x in fnames if not os.path.exists(os.path.join(output_dir, x))]
    print(f"Processing {len(fnames)} files...")
    
    tasks = []
    for fname in fnames:
        with open(os.path.join(request_dir, fname), "r") as f:
            data = json.load(f)
        
        tasks.append(
            asyncio.ensure_future(
                predict_with_semaphore(sem, fname, data, output_dir)
            )
        )
    results = await tqdm_asyncio.gather(*tasks)
    # for fname, result in results:
    #     with open(os.path.join(output_dir, fname), "w") as f:
    #         json.dump(result.model_dump(), f, indent=2)
    print(f"Processed {len(results)} files.")

if __name__ ==  '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())