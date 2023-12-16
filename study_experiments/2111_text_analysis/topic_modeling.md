# Topic Modeling
Extracting topic from a corpus.[4]
Recognizing words from topics present in document or the corpus of data.[3]

## Topic Vectors
Can represent meaning of words & entire documents
(전체 document의 의미를 표현)
한번 벡터화 되어 있는 데이터에 대해서 topic vector를 구축한다

* semantic search: search for documents based on their meaning
* identify(extract) words that best represent topic -> can be used for summarization
* document similarity

## Topic Modeling Algorithms
### 1. LSA (Latent Semantic Analysis)
Uses SVD (Singular Value Decomposition) to find combinations of words that are responsible, for the <b>biggest variation</b> in the data. Only have to retain high-variance dimensions. Each of these dimensions becomes "topic".

LSA learns latent topics by performing matrix decomposition on document-term matrix using SVD[5].

![LSA Example[5]](figs/topic_lsa.jpeg)
A = U x S x V^T
* A: Term-Document Matrix (row: Term(word), col: Document)
* U: Word Assignment to Topics
    * Each row is word vector
* S: Topic Importance
* V^T: Topic Distribution Across Documnets
    * Each column is document vector


Use truncated SVD to select number of topics

### 2. PCA (Principal Component Analysis)

### 3. LDA (Latent Dirichlet Allocation)
특정 토픽에 특정 단어가 나타날 확률 계산

## References
[1] SKKU NLP Course - Prof. Yun-Gyung Cheong<br>
[2] https://towardsdatascience.com/document-summarization-using-latent-semantic-indexing-b747ef2d2af6<br>
[3] https://www.analyticsvidhya.com/blog/2021/05/topic-modelling-in-natural-language-processing/<br>
[4] https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/07/09/lda/<br>
[5] https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python