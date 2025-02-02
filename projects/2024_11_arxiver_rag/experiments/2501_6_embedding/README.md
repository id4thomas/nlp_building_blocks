# embedding models experiments
- 텍스트, 이미지를 함께 임베딩 할 수 있는 방법들을 살펴본다 (2025.01 기준)

## 조사
### 모델 옵션
임베딩 용으로 학습된 모델 / VLM → 임베딩처럼 활용 2가지 옵션 체크

- nvidia/mm-embed (nc, 8.2b)
    - https://huggingface.co/nvidia/MM-Embed
    - (img) llava-1.6-mistral-7b + (text) nv-embed
- jinaai/jina-clip-vs (nc)
    - https://huggingface.co/jinaai/jina-clip-v2
- e5-v (라이센스x, 8.3b)
    - `LlavaNextForConditionalGeneration`
    - https://huggingface.co/papers/2407.12580
    - https://huggingface.co/royokong/e5-v
    - https://huggingface.co/lmms-lab/llama3-llava-next-8b 기반
- VisRAG-Ret (Apache, 3.4b)
    - https://huggingface.co/openbmb/VisRAG-Ret
    - 기반 모델: https://huggingface.co/openbmb/MiniCPM-V-2
        - MiniCPMV 기반 - (vis) SigLIP + (text) MiniCPM-2B
        - 자체 모델 클래스 정의 (VisRAG_Ret): forward안에 자체로 쓰레드 처리 있음 → MiniCPMV로 로드 가능한지 체크 필요
- baai/bge-visualized
    - https://huggingface.co/BAAI/bge-visualized
    - 커스텀 패키지

### 모델 서빙 프레임워크
#### vllm (v0.7.0 기준)
- 멀티모달 관련
    - 이미지 ‘임베딩’을 모델에 입력 https://docs.vllm.ai/en/stable/serving/multimodal_inputs.html#embedding
    - 지원 모델 리스트: https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models
- 임베딩 시도하더라도 OpenAI 상으로는 지원 x

#### llama.cpp
- embedding 가이드
    - https://github.com/ggerganov/llama.cpp/discussions/7712
- multimodal 기능 자체가 삭제된 상황 (지원하는 모델은 일부 별도로 존재)
    - https://github.com/ggerganov/llama.cpp/issues/8010

#### text-embedding-inference
- issue만 열려있는 상황 (https://github.com/huggingface/text-embeddings-inference/issues/333)

#### ollama

- issue만 열려있는 상황 (https://github.com/ollama/ollama/issues/5304)

#### 유료 API 서비스
- jinaai - embedding API에 멀티모달 지원
    - `https://api.jina.ai/v1/embeddings`
- voyage AI - multimodal embedding 전용 모델
    * https://blog.voyageai.com/2024/11/12/voyage-multimodal-3/

### rag 프레임워크
- langchain, llama-index에서 멀티모달 검색 현황 확인

#### langchain
- The current embedding interface used in LangChain is optimized entirely for text-based data, and will not work with multimodal data.
    - https://python.langchain.com/docs/concepts/multimodality/
- Chromadb는 기능이 별도로 있어는 보임
    - https://python.langchain.com/v0.1/docs/templates/rag-chroma-multi-modal/

#### llama-index
- multimodal-rag (2023.11 작성): 텍스트, 이미지 각각 retrieval 방법. 별도 collection 구성
    - https://www.llamaindex.ai/blog/multi-modal-rag-621de7525fea
    - https://docs.llamaindex.ai/en/stable/examples/multi_modal/multi_modal_retrieval/

## Expriments
### 1_multimodal_embedding_vllm
* use vllm to infer multimodal embeddings
* qwen2-vl-2b 테스트

VLM 모델을 ‘embed’ 태스크 옵션으로 로드해서 추론 가능
```python
llm = LLM(
    model=model_dir,
    max_model_len=8192,
    limit_mm_per_prompt={"image": 4},
    task="embed",
)
text = ":텍스트1 이미지 1: <|image_pad|> 이미지2: <|image_pad|>"
image1 = PIL.Image.open("image.png")
prompt = {
    "prompt": text,
    "multi_modal_data": {
        "image": [image1, image1]
    },
}
outputs = llm.embed(prompt)
embedding = outputs[0].outputs.embedding # list[float] embedding
```

- ‘prompt’ 텍스트의 ‘hf_processor.image_token’ 토큰을 주어진 ‘multi_modal_data’의 이미지로 교체
    - 순서 처리는 확인 필요

