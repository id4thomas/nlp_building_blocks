# langchain + faiss
* memory usage
    * https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-not-hnswm-or-ivf1024pqnx4fsrflat
    * all Faiss indexes are stored in RAM
    * when usign 'hnsw': `(d * 4 + M * 2 * 4) bytes` per vector
        * `M`: links per vector
        * `efSearch`: speed-accuracy tradeoff
        * gpu not supported

## Envs
