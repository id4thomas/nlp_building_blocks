# 2024.11 - arxiver RAG
* `neuralwork/arxiver` 데이터 기반으로 RAG 전처리 → 인덱싱 → 검색 파이프라인을 직접 구축해본다
    * https://huggingface.co/datasets/neuralwork/arxiver

Folder Tree
```
.
├── README.md
└── experiments # 작은 단위의 기능 테스트 실험들
```

# Experiments
| experiment | description |
| --- | --- |
| 2410_1_vlm_test | colqwen->vlm POC test |
| 2411_1_semantic_splitter | semantic chunking test |
| 2411_2_section_splitter | developing splitter module |
| 2412_1_section_chunking | developing chunking module |