# 241027 - image_based document understanding
## Dataset
* arxiver - processed arxiv dataset
    * https://huggingface.co/datasets/neuralwork/arxiver
```
└── arxiver
    ├── data
    │   └── train.parquet
    └── README.md
```
* Use arxiv-downloader to get PDF
    * https://github.com/braun-steven/arxiv-downloader
* Convert pdf->img via pdf2image


## Experiments
* 0_check_dataset
    * Check downloaded arxiver dataset
    * use arxiv-downloader to download pdf file
    * pdf->img with `pdf2img.convert_from_path`
* 1_qwen_vl_test
    * Test Image based chat Q&A with `Qwen2-VL-7B-Instruct`
    * Use openai client + vllm-openai for inference
* 2_colqwen_retrieval_test
    * Test query-Image retrieval with `colqwen2-v0.1`
    * use colpali_engine package [link](https://github.com/illuin-tech/colpali)
