# LLM Throughput Calculation

## Test Settings
Testcase
- Prompt 4096 Tokens, Generation 1024 Tokens


## Usage Example
### 1. Define config
Modify the following Values
- Served Endpoint info: `base-url`, `env.OPENAI_API_KEY`
- Model:
  - `model`: model name recognized by the endpoint
  - `tokenizer`: host directory path of the model's tokenizer
- Dataset:
  - `num-prompts`: total number of requests to process
  - `input-len`: input prompt token size
  - `output-len`: number of tokens to generate per request
- Concurrency: `max-concurrency` client concurrency size


Example
```
backend: openai-chat
base-url: http://localhost:901
endpoint: /v1/chat/completions

model: Qwen3.6-35B-A3B
tokenizer: /home/yrlab/models/llm/weights_local/Qwen3.6-35B-A3B-FP8

dataset-name: random
num-prompts: 100
input-len: 4096
output-len: 1024

max-concurrency: 8
request-rate: inf

env:
  OPENAI_API_KEY: sk-123
```

### 2. Run vllm bench
```
# ex. ./run_bench example
./run_bench {CONFIG_NAME}
```

## Results

DGX Spark (GB10)

| Model | Dtype | Concurrency | RPS | Time per 1K Req | Output TPS | Total TPS |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B | BF16 | 1 | 0.029 | 35044s (584.07 Min) | 29 | 146 |
| Qwen3.5-35B-A3B | BF16 | 4 | 0.069 | 14589s (243.15 Min) | 70 | 350 |
| Qwen3.5-35B-A3B | BF16 | 8 | 0.086 | 11566s (192.77 Min) | 88 | 442 |
| Qwen3.6-35B-A3B | FP8 | 1 | 0.046 | 21651s (360.85 Min) | 47 | 236 |
| Qwen3.6-35B-A3B | FP8 | 4 | 0.100 | 9975s (166.25 Min) | 102 | 513 |
| Qwen3.6-35B-A3B | FP8 | 8 | 0.111 | 8976s (149.6 Min) | 114 | 570 |
