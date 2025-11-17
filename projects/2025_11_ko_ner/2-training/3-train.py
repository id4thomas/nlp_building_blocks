import os
from datasets import load_dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

model_name = "skt/A.X-Encoder-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 1. Load Data
labels = ["O"]
ENTITY_TYPES = ["PS", "LC", "OG", "DT", "TI", "QT"]

for ent in ENTITY_TYPES:
    labels.append(f"B-{ent}")
    labels.append(f"I-{ent}")

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_and_align_labels(examples):
    """
    examples["text"]: 문자열 리스트
    examples["entities"]: 엔티티 스팬 리스트의 리스트
        하나의 문장에 대해 [{"start":..., "end":..., "label":...}, ...]
    => tokenized input + BIO 라벨(id)을 반환
    """
    texts = examples["text"]
    all_entities = examples["entities"]

    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        return_offsets_mapping=True
    )

    all_labels = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        entities = all_entities[i]
        # 엔티티 스팬들을 빠르게 조회하기 위해
        # (start, end, label) 형태로 정렬
        spans = [(e["start"], e["end"], e["label"]) for e in entities]

        labels_ids = []
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                # special token ([CLS], [SEP]) 등
                labels_ids.append(-100)
                continue

            token_label = "O"

            # 이 토큰이 포함되는 엔티티가 있는지 확인
            for ent_start, ent_end, ent_label in spans:
                # (토큰의 span)와 (엔티티 span)이 겹치면 엔티티로 간주
                if not (end <= ent_start or start >= ent_end):
                    # 겹친다는 뜻
                    # 해당 엔티티 내에서의 위치에 따라 B- / I- 결정
                    if start == ent_start:
                        token_label = f"B-{ent_label}"
                    else:
                        token_label = f"I-{ent_label}"
                    break  # 가장 먼저 매칭된 엔티티 사용

            labels_ids.append(label2id[token_label] if token_label in label2id else label2id["O"])

        all_labels.append(labels_ids)

    # offset_mapping은 모델에 넣을 필요 없으므로 제거
    tokenized.pop("offset_mapping")

    tokenized["labels"] = all_labels
    return tokenized

data_dir = "./data/run1"
data_files = {
    "train": os.path.join(data_dir, "train.parquet"),
    "val": os.path.join(data_dir, "val.parquet"),
    "test": os.path.join(data_dir, "test.parquet"),
}

ds = load_dataset('parquet', data_files=data_files)
train_ds = ds['train']
train_ds = train_ds.map(
    encode_and_align_labels,
    batched=True,
    remove_columns=train_ds.column_names  # text, entities 제거하고 인풋/라벨만 남김
)
val_ds = ds['val']
val_ds = val_ds.map(
    encode_and_align_labels,
    batched=True,
    remove_columns=val_ds.column_names  # text, entities 제거하고 인풋/라벨만 남김
)

# 2. Prepare Model

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
base_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config,
    dtype=torch.bfloat16
)
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,   # 토큰 분류 작업
    r=8,                            # 랭크
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules = [
    "Wi",
    "Wqkv",
    "dense",
    "Wo"
  ]
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to("mps")

# metrics 계산 함수
def compute_metrics(p):
    """
    p.predictions: (batch, seq_len, num_labels)
    p.label_ids: (batch, seq_len)
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    # -100 라벨 제외
    true_labels = []
    true_preds = []

    for pred, lab in zip(predictions, labels):
        for p_i, l_i in zip(pred, lab):
            if l_i == -100:
                continue
            true_preds.append(p_i)
            true_labels.append(l_i)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="macro"
    )
    acc = accuracy_score(true_labels, true_preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./output/test",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()