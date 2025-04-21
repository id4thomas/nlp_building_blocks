import json
from typing import List, Dict, Any

import torch.utils.data as data
import pandas as pd

SYSTEM_MESSAGE = '''You are building an emotion‑prediction component that, given a character name and a brief situational description, must identify the character’s emotional state and the reason for it.
When given an input of the form:
```
Character:  {character}
Source: {source}
```
your job is to:

1. **Map the Emotion:**  
   - Choose among the 8 primary emotions from Plutchik’s Wheel (joy, trust, fear, surprise, sadness, disgust, anger, anticipation).  
   - For each, assign one of: `"na"` (not applicable), `"low"`, `"medium"`, or `"high"`.

2. **Write a Reason:**  
   - Provide a single sentence explaining why the character feels as you’ve labeled.

## Map the Emotion: Map the target emotion onto the 8 primary emotions from Plutchik’s Wheel
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

Return in the following JSON format (no extra keys, no explanation outside the JSON)
{
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
}
Only return the JSON'''

USER_TEMPLATE = '''Source: {source}
Character: {character}'''


class MessagesDataset(data.Dataset):
    '''
    data_dir: data in tsv format
    '''
    def __init__(self, data_dir: str, tokenizer, tokenization_config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.tokenization_config = tokenization_config

        df = pd.read_csv(data_dir, sep = "\t")
        self.sources = df.apply(self._make_source, axis = 1)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(self.sources[idx], return_tensors = "pt", **self.tokenization_config)
        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        encoded = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return encoded
    
    def _make_source(self, row):
        user_message = USER_TEMPLATE.format(
            source=row['source'],
            character=row['character']
        )
        assistant_message = json.dumps(
            {
                "emotion": {
                    "joy": row['joy'],
                    "trust": row['trust'],
                    "fear": row['fear'],
                    "surprise": row['surprise'],
                    "sadness": row['sadness'],
                    "disgust": row['disgust'],
                    "anger": row['anger'],
                    "anticipation": row['anticipation']
                },
                "reason": row['reason']
            }
        )

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        source = self.tokenizer.apply_chat_template(messages, tokenize = False)
        return source
