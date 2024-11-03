# 241103 - Semantic Splitter test
* Retrieval용 청킹 방식 중 Semantic Chunking에 대해 파악해본다
* semantic-chunkers 패키지로 테스트를 진행해본다

## Dataset
* arxiver - processed arxiv dataset
    * https://huggingface.co/datasets/neuralwork/arxiver
```
└── arxiver
    ├── data
    │   └── train.parquet
    └── README.md
```

## semantic-chunkers (aurelio.ai)
https://github.com/aurelio-labs/semantic-chunkers

3 종류의 chunker 제공
- Statistical Chunker: identifying shifts in semantic similarity
- ConsecutiveChunker: divides content into fixed-size chunks based on set word or character limits
- CumulativeChunker: accumulating content until reaching a similarity threshold

Statistical Chunker
- uses a varying similarity threshold to identify more dynamic and local similarity splits

Cumulative Chunker
- more compute intensive process, but can often provide more stable results as it is more noise resistant
- *very expensive* in both time and (if using APIs) money