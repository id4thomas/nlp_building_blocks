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
* VADER
* TextBlob: 
    * sentiment polarity (+,-)
    * subjectivity: how somesome's judgement is shaped by personal opinions

## References
[1] https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools<br>
[2] https://skyjwoo.tistory.com/entry/자연어-처리-EDAExploratory-Data-Analysis<br>
[3] https://towardsdatascience.com/getting-started-with-text-analysis-in-python-ca13590eb4f7<br>
[4] https://wikidocs.net/33661<br>
[5] https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html<br>
[6] https://www.kaggle.com/abhilash1910/tweet-analysis-eda-cleaning-tsne-glove-tf<br>