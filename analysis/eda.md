# EDA (Exploratory Data Analysis)

Mainly studying methods in [1].

* [Text Statistics](#text-statistics)
* Ngram Exploration
* Topic Modeling
* Wordcloud
* [Sentiment Anslysis](#sentiment-analysis)
* Named Entity Recognition
* Part of Speech Tagging
* Textstat library: readability Score

## Text Statistics
Histogram based visualization

### word frequency anslysis

* CountVectorizer: Convert a collection of text documents to a matrix of token counts. (Term-Document matrix)
    * Uses lowercase only by default
    * default configuration tokenizes the string by extracting words of at least 2 letters
    * [CountVectorizer Example](eda_text_statistics.ipynb)

* Counter: use most_common() function
    

### sentence length analysis
### average word length analysis

## Sentiment Analysis
### VADER (Valence Aware Dictionary for Sentiment Reasoning)
Rule-based model for General Sentiment Analysis
#### 5 Generalizable Heuristics
1. Punctuation
    * '!' increases magnitude of intensity without modifying semantic orientation
2. Capitalization
    * All-Caps representation of Sentiment-relevant word increases magnitude of intensity without modifying semantic orientation
3. Degree Modifiers (Intensifiers, Booster Words, Degree Adverbs)
    * Increase/Decrease Intensity
4. Contrastive Conjunction 'but'
    * Shift in sentiment polarity
    * Text following the conjunction is dominant
5. Examining Tri-gram preceding Sentiment-laden Lexical Feature
    * Carch nearly 90% of cases where negation flips polarity

Use nltk.sentiment.vader.SentimentIntensityAnalyzer to get sentiment intensity scores. <i>polarity_scores(sent)</i> function returns dictionary with keys ['neg','neu','pos','compound']. Compound value is computed by normalizing the sum of neg,neu,pos scores. <br>
Experiment in [notebook](eda_sentiment.ipynb)
    
### TextBlob: 
* sentiment polarity (+,-)
* subjectivity: how somesome's judgement is shaped by personal opinions

## References
[1] https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools<br>
[2] https://skyjwoo.tistory.com/entry/자연어-처리-EDAExploratory-Data-Analysis<br>
[3] https://towardsdatascience.com/getting-started-with-text-analysis-in-python-ca13590eb4f7<br>
[4] https://wikidocs.net/33661<br>
[5] https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html<br>
[6] https://www.kaggle.com/abhilash1910/tweet-analysis-eda-cleaning-tsne-glove-tf<br>
[7] https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664