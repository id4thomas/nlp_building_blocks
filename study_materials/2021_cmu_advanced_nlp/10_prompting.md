# 10 - Representation 3 - Prompting + Pre-training

## Table of Contents
* [Four Paradigms of NLP Technical Development](#Four-Paradigms-of-NLP-Technical-Development)
    * Feature Engineering
    * Architecture Engineering
    * Objective Engineering
    * Prompt Engineering
* [Prompting](#Prompting)
    * [General Workflow of Prompting](#General-Workflow-of-Prompting)
        * Prompt Addition
        * Answer Prediction
        * Answer-Label Mapping
    * [Design Considerations for Prompting](#Design-Considerations-for-Prompting)
        * Pre-trained Model Choice
        * Prompt Engineering
        * Answer Engineering
        * Expanding the Paradigm
        * Prompt-based Training Strategies


## Four Paradigms of NLP Technical Development
### Feature Engineering
* Paradigm: Fully Supervised Learning <b>(Non-neural)</b>
* Characteristics:
    * Non-neural ML Models
    * Require manually defined feature extraction
        * Sentence -> word/bigram (Manual Features) -> CLF
* Representative Work:
    * SVM
    * Conditional Random Fields (CRF) for sequence labeling
### Architecture Engineering
* Paradigm: Fully Supervised Learning <b>(Neural nets)</b>
* Characteristics:
    * Rely on neural networks
    * Don't need to manually define features
    * But should modify network structure
        * Ex. LSTM for long-term dependency, CNN for local
    * Sometimes use pre-training but for shallow features such as embeddings (Word2Vec등)
* Representative Work:
    * CNN for Text Classification
### Objective Engineering
* Paradigm: Pre-train, Fine-tune
* Characteristics:
    * Pre-trained LMs used as initialization of full model
        * Both shallow and deep features
    * Less work on architecture design but <b>engineer objective function</b>
        * Few extra features you add on top
* Typical Work:
    * Bert -> Finetune
### Prompt Engineering
* Paradigm: Pre-train, Prompt, Predict
    * Prompt: Specify what you want the model to do
* Characteristics:
    * NLP tasks <b>modeled entirely by relying on LMs</b>
        * Don't even do further training (웨이트 고정)
    * Tasks of shallow & deep feature extraction and prediction are all given to LM
    * Engineering of prompts is required
* Representative Work:
    * GPT3

## Prompting
Encourage pre-trained model to make particular predictions by providing prompt
간단하게는 Text 형태의 prompt

Ex. MLM에 인풋을 "CMU is located in [MASK]." 로 주고 모델이 맞추기

## General Workflow of Prompting
### Prompt Addition
### Answer Prediction
### Answer-Label Mapping

## Design Considerations for Prompting
* Pre-trained Model Choice
* Prompt Engineering
* Answer Engineering
* Expanding the Paradigm
* Prompt-based Training Strategies

### Pre-trained Model Choice
* 어떤 종류 모델 고르냐 따라 prompt 종류가 달라짐

### Prompt Engineering
* Prompt Template Engineering
    * Hand-crafted
    * Automated Search
        * Search in Discrete Space
            * Prompt Mining
            * Prompt Paraphrasing
            * Gradient-based Search
        * Search in Continuous Space
            * Prompt/Prefix Tuning
### Answer Engineering
* Answer Search
    * Hand-crafted
        * Infinite answer space: Free-form
        * Finite answer space: Good for CLF
    * Automated Search
        * Discrete Space
            * Answer Paraphrasing
            * Prune-then-Search
            * Label Decomposition
        * Continuous Space
            *  Virtual Class Token & Optimize token embedding

### Expanding the Paradigm
* Multi-Prompt Learning
    * Prompt Ensemble
    * Prompt Augmentation
    * Prompt Composition
    * Prompt Decomposition
    * Prompt Sharing

### Prompt-based Training Strategies
* Data Perspective
    * Zero-shot
    * Few-shot
    * Full-data
* Parameter Perspective
    * 어떤 파라미터를 Tune하는지 Fix 하는지
    * Proptless Fine-tuning
    * Tuning-free Prompting
    * Fixed-LM Prompt Tuning
    * Fixed-Prompt LM Tuning
    * Prompt+LM Fine-Tuning
