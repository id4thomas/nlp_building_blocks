import asyncio
from dotenv import load_dotenv
import os
import yaml
from typing import List

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import json
from pathlib import Path

from schema import NerResult

# ------------------------
# Load env
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_REASOINING_EFFORT = os.getenv("OPENAI_REASOINING_EFFORT")

PROMPT_NAME = os.getenv("PROMPT_NAME")
print("Prompt:", PROMPT_NAME)

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

if OPENAI_API_KEY is None or OPENAI_BASE_URL is None:
    raise RuntimeError("OPENAI_API_KEY or OPENAI_BASE_URL variable missing")


# ------------------------
# YAML loader
# ------------------------
def load_prompt_template(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        tpl = yaml.safe_load(f)
    return tpl


# ------------------------
# Call OpenAI for NER + Save outputs
# ------------------------
async def call_openai_for_ner(instruction: str, prompt: str, save_base_path: str, doc_id: str) -> NerResult:
    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": prompt}
    ]

    response = await client.responses.parse(
        model=OPENAI_MODEL,
        input=messages,
        text_format=NerResult,
        reasoning={"effort": OPENAI_REASOINING_EFFORT}
    )

    # print(response.usage)

    # ---------- Save raw response ----------
    response_dir = Path(save_base_path) / "response"
    response_dir.mkdir(parents=True, exist_ok=True)
    with open(response_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)

    # ---------- Save parsed output ----------
    generated_dir = Path(save_base_path) / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    with open(generated_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(response.output_parsed.model_dump(), f, ensure_ascii=False, indent=2)

    return response.output_parsed


async def process_texts(texts: List[str], yaml_template_path: str, save_base_path: str, doc_ids: List[str]) -> pd.DataFrame:
    tpl = load_prompt_template(yaml_template_path)
    instruction = tpl['prompt']['developer']

    semaphore = asyncio.Semaphore(32)  # limit concurrency to 32

    async def safe_call(instruction, prompt, save_base_path, doc_id):
        """Wrapper that retries up to 5 times and skips on persistent failure."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                async with semaphore:
                    return await call_openai_for_ner(instruction, prompt, save_base_path, doc_id)
            except Exception as e:
                print(f"[ERROR] doc_id={doc_id} attempt={attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    print(f"[SKIP] doc_id={doc_id} after {max_retries} failed attempts.")
                    return None
                # simple backoff: 1, 2, 3, 4... seconds
                await asyncio.sleep(attempt)

    tasks = []
    base_path = Path(save_base_path)
    response_base = base_path / "response"

    for txt, doc_id in zip(texts, doc_ids):
        # --- Skip if response JSON already exists ---
        response_path = response_base / f"{doc_id}.json"
        if response_path.exists():
            # print(f"[SKIP EXISTING] doc_id={doc_id} (response JSON already exists)")
            continue

        prompt = tpl["prompt"]["user1"].replace("{TEXT}", txt)
        tasks.append(
            safe_call(instruction, prompt, save_base_path, doc_id)
        )

    if not tasks:
        print("No tasks to run (all doc_ids already processed).")
        return pd.DataFrame([])

    results = await tqdm.gather(*tasks)

    # Build DataFrame (skip failed ones)
    records = []
    for ner_res in results:
        if ner_res is None:
            continue
        for ent in ner_res.entities:
            records.append({
                "tagged_text": ner_res.tagged_text,
                "entity_value": ent.value,
                "entity_label": ent.label.value,
            })

    return pd.DataFrame(records)


async def main():
    fname = "NEWSPAPER_2022_2"
    dataset_name = "NIKL_NEWSPAPER_2023_CSV"
    # fname = "NEWSPAPER_2022_1"
    # sample_name = "sample1"
    # sample_name = "sample2"
    # fname = "NEWSPAPER_2022_2"
    # sample_name = "sample3"
    fname = "NEWSPAPER_2022_3"
    sample_name = "sample4"

    df = pd.read_parquet(f"data/{dataset_name}-{fname}-{sample_name}.parquet")
    # df = df.iloc[:5]
    print(f"{dataset_name}-{fname}-{sample_name}", df.shape, df.columns)

    # Required column: "doc_id", "text" (adjust if different)
    texts = df["text"].tolist()
    doc_ids = df["doc_id"].astype(str).tolist()

    prompt_path = f"prompt/{PROMPT_NAME}.yaml"
    save_base_path = f"results/{dataset_name}-{fname}-{sample_name}"
    # Ensure base directory exists
    Path(save_base_path).mkdir(parents=True, exist_ok=True)

    result_df = await process_texts(texts, prompt_path, save_base_path, doc_ids)

    # If nothing new was processed, result_df may be empty
    if not result_df.empty:
        result_df.to_parquet(f"{save_base_path}/entities.parquet", index=False)#, encoding="utf-8")
    else:
        print("No new entities to save (result_df is empty).")


if __name__ == "__main__":
    asyncio.run(main())