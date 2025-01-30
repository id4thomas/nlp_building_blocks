# langchain + faiss
* memory usage
    * https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-not-hnswm-or-ivf1024pqnx4fsrflat
    * all Faiss indexes are stored in RAM
    * when usign 'hnsw': `(d * 4 + M * 2 * 4) bytes` per vector
        * `M`: links per vector
        * `efSearch`: speed-accuracy tradeoff
        * gpu not supported

| **Index Type**         | **Best For**                                                                 | **Advantages**                    | **Considerations**                      |
|-------------------------|-----------------------------------------------------------------------------|------------------------------------|------------------------------------------|
| **Flat**               | Small datasets; exact nearest neighbor searches                              | High accuracy (exact search)       | Memory-intensive for large datasets      |
| **HNSW**               | Small to medium datasets (<1M vectors); high recall and fast searches        | Fast, high recall                  | Requires more memory                     |
| **IVF Flat**           | Medium to large datasets (≥1M vectors); approximate nearest neighbor search  | Balanced speed and memory usage    | Slight reduction in accuracy             |
| **IVF-PQ**             | Very large datasets; memory-constrained scenarios                            | Low memory usage                   | Lower precision due to quantization      |
| **PQ**                 | Extremely large datasets with tight memory limits                           | High memory efficiency             | Requires tuning for good performance     |

## Envs
