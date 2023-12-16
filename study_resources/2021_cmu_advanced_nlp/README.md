# CMU Advanced NLP CS 11-711 (Fall-2021)
* [Cource Materials](http://phontron.com/class/anlp2021/schedule/class-introduction.html)
* [Lectures](https://www.youtube.com/watch?v=pGC-QiNbuwA)

## Schedule
* 6. Modeling 2 - Conditioned Generation
* 10. Representation 3 - Prompting + Pre-training [[summary]](#10.-Representation-3:-Prompting-+-Pre-training) [[link]](./10_prompting.md)
* 14. Analysis 2 - Syntactic Parsing 1 [[summary]](#14.-Analysis-2:-Syntactic-Parsing-1) [[link]](./14_syntactic_parsing1.md)
* 15. Analysis 3 - Syntactic Parsing 2
* 16. Analysis 4 - Semantic Parsing 

## Summarization
### 6. Modeling 2: Conditioned Generation
* Formulating and Modeling
* Methods of Generation: P(Y|X) 모델을 가지고 어떻게 문장을 생성할지
    * Sampling: Ancestral Sampling (Randomly Generate One-by-One)
    * Argmax: Generate sentence with highest score
        * Greedy
        * Beam
* Model Ensembling
    * Linear Interpolation: Weighted Average of M model probabilities
    * Log-linear Interpolation: weighted combination of log probabilites, normalize Softmax(Interpolation coefficient for model m * Log prob of model m)
    * Parameter Averaging
* Case Studies in Conditional Language Modeling
    * Level of Constraint of Output
    * Controlled Generation
* How do we evaluate
    * Human Evaluation
    * Compare with Reference (Quality Estimation)
        * BLEU,ROUGE
        * Embedding-based Metrics: based on Neural Models
    * Perplexity: calculate without generation

### 10. Representation 3: Prompting + Pre-training
* Four Paradigms of NLP Technical Development
* Prompting
    * General Workflow of Prompting
        * Prompt Addition / Answer Prediction / Answer-Label Mapping
    * Design Considerations for Prompting
        * Pre-trained Model Choice
        * Prompt Engineering / Answer Engineering
            * Manual, Automatic Search (Discrete/Continuos)
        * Expanding the Paradigm
            * Multi-Prompt Learning: Ensemble, Augmentation, Composition, Decomposition, Sharing
        * Promp-based Training Strategies
            * Data Perspective: Zero-/Few-Shot, Full-data
            * Parameter Perspective: 어떤 파라미터를 Tune 하는지 Fix 하는지

### 14. Analysis 2: Syntactic Parsing 1
* Syntacting Parsing
    * POS
    * Parsing
        * Constituency Trees
        * Dependency Trees
    * Context-Free Grammars (CFGs)

### 15. Analysis 3: Syntactic Parsing 2
### 16. Analysis 4: Semantic Parsing 

