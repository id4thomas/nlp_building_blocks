# nlp_building_blocks
Studying NLP one by one

## Table of Contents
* [Preprocessing](./preprocessing/README.md)
    * [Huggingface Tokenizer Analysis](./preprocessing/tokenizer_huggingface.md)
        * [Examples](./preprocessing/tokenize_huggingface_examples.ipynb)
* [Analysis](./analysis/README.md)
    * [EDA (Exploratory Data Analysis)](./analysis/eda.md)
        * Text Statistics
            * n-gram Exploration
            * [Topic Modeling](./analysis/topic_modeling.md)
            <!-- * Wordcloud -->
            * [Examples](./analysis/eda_text_statistics.ipynb)
        * Sentiment Analysis
            * VADER
            * TextBlob
            * Flair
            * [Examples](./analysis/eda_sentiment.ipynb)
* [Vectorization](./vectorization/README.md)
    * [Count Based Representations](./vectorization/count_based_representations.md)
        * Bag-of-Words (Binary,Count Values)
        * TF-IDF
        * [Examples](./vectorization/count_based_representations.ipynb) 
* [Transformer](./transformer/README.md)
    * Studying and implementing the Transformer Architecture
    * [Transformer Implementation Details](./transformer/implementation/README.md)
* [Evaluation](./evaluation/README.md)
    * Methods of evaluating NLP Models, generated results
    * Reference-based Metrics
        * [BLEU](./evaluation/bleu.md)
        * [ROUGE](./evaluation/rouge.md)
            * [Examples](./evaluation/rouge.ipynb)
    * Transfer-learning based Metrics
        *[BLEURT]