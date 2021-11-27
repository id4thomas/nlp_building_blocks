# ROUGE
Analyzing ROUGE
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is recall oriented measure for evaluating text generation systems<br>
Usually used for evaluating <b>text summarization & machine translation</b><br>
Compares model generated sentence with reference sentence.

## Variations
* ROUGE-N
* ROUGE-L
* ROUGE-S


## Implementation Analysis
### ROUGE Implementations
* [files2rouge](https://github.com/pltrdy/files2rouge) (Official Rouge) 
    * Also provides python implementation ([rouge](https://github.com/pltrdy/rouge) pip install rouge)
* [rouge-metric](https://github.com/li-plus/rouge-metric)
    * A Python wrapper of the official ROUGE-1.5.5.pl
    * (pip install rouge-metric)
* Huggingface datasets metric [[link](https://huggingface.co/metrics/rouge)]

Examples for each package at [rouge.ipynb](./rouge.ipynb)

### Step-by-Step
1. Split into sentences

## References
[1] https://huffon.github.io/2019/12/07/rouge/<br>
[2] https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460<br>
[3] https://github.com/pltrdy/rouge<br>
[4] https://github.com/li-plus/rouge-metric<br>
[5] https://github.com/pltrdy/rouge/issues/2<br>
[6] https://github.com/google/seq2seq/issues/89<br>