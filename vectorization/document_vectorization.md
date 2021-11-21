# Document Vectorization

* [Simple Methods](#simple-methods)
    * [Binary: OneHot](#onehot\-vector)
    * [Count Vector](#count-vector)
    * [TF-IDF](#tf\-idf)


# Simple Methods

## Onehot Vector

CountVectorizer의 binary=True 옵션 이용

True일 경우 non-zero count가 1로 세팅됨 (존재 유무만 리포팅)

```python
Train Vectors
This is the first document. [0 1 1 1 0 0 1 0 1]
and:0	document:1	first:1	is:1	one:0	second:0	the:1	third:0	this:1	

This document is the second document. [0 1 0 1 0 1 1 0 1]
and:0	document:1	first:0	is:1	one:0	second:1	the:1	third:0	this:1	

And this is the third one. [1 0 0 1 1 0 1 1 1]
and:1	document:0	first:0	is:1	one:1	second:0	the:1	third:1	this:1	

Is this the first document? [0 1 1 1 0 0 1 0 1]
and:0	document:1	first:1	is:1	one:0	second:0	the:1	third:0	this:1	

Test Vectors
This is the fourth document. [0 1 0 1 0 0 1 0 1]
and:0	document:1	first:0	is:1	one:0	second:0	the:1	third:0	this:1	

This document is the one. [0 1 0 1 1 0 1 0 1]
and:0	document:1	first:0	is:1	one:1	second:0	the:1	third:0	this:1	

This document is new. [0 1 0 1 0 0 0 0 1]
and:0	document:1	first:0	is:1	one:0	second:0	the:0	third:0	this:1
```

## Count Vector

문서를 token count matrix로 변환

기본 토큰 패턴: r"(?u)\\b\\w\\w+\\b"

Select tokens of 2 or more alphanumeric characters

Punctuation is completely ignored, always treated as token separator

sklearn의 CountVectorizer 분석 (1.0.1 Version 기준)

[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

- 참고 파라미터
    - **input: *{‘filename’, ‘file’, ‘content’}, default=’content’***
    - **lowercase: *bool, default=True***
        - Default가 lowercase화 참고
    - **tokenizer: *callable, default=None***
        - analyzer=="word"일 경우에만 해당, 기본 토큰화는 build_tokenizer()로 테스트 가능
        - 기본 패턴: r"(?u)\\b\\w\\w+\\b"
        - **token_pattern** 파라미터에서 regex 표현 직접 커스텀 가능
    - **stop_words: *{‘english’}, list, default=None***
        - 기본 'english' 옵션이 문제가 있으니 alternative 추천한다
    - **ngram_range: *tuple (min_n, max_n), default=(1, 1)***
        - (1,2) 면 unigram, bigram 모두
        - (2,2) 면 bigram만
        - **analyzer 파라미터로 word n-gram일지 character n-gram일지 선택 가능**
    - **analyzer: *{‘word’, ‘char’, ‘char_wb’} or callable, default=’word’***
        - character_**wb**는 **word boundary 내에서만 n-gram 만든다** (edges of words are padded with space**)**
    - **max_df: *float in range [0.0, 1.0] or int, default=1.0***
        - **ignore** terms that have a document frequency strictly **higher than the given threshold**
        - float면 proportion of document, int면 absolute counts로 처리된다
        - **vocabulary**를 따로 제공하면 쓰이지 않음
    - **min_df: *float in range [0.0, 1.0] or int, default=1***
        - max_df와 마찬가지로 이건 lower bound
    - **max_features: *int, default=None***
        - top max_features로 vocabulary 만들것인지
    - **vocabulary: *Mapping or iterable, default=None***
        - key-value (dict)같은 Mapping이나, iterable over terms 형태로 제공
        - should not be repeated, should not have any gap
    - **binary: *bool, default=False***
        - True일 경우 non-zero count가 1로 세팅됨 (존재 유무만 리포팅)

- 참고 메서드
    - build_tokenizer()
        - Tokenize하는 함수 불러옴
    - get_feature_names_out([input_features])
        - [input_features]는 api 일관성 위한것이고 필요 x
        - 사용된 feature names (vocabulary)를 가져옴
    - inverse_transform(X)
        - terms per document로 변환함

```python
Train Vectors
This is the first document. [0 1 1 1 0 0 1 0 1]
and:0	document:1	first:1	is:1	one:0	second:0	the:1	third:0	this:1	

This document is the second document. [0 2 0 1 0 1 1 0 1]
and:0	document:2	first:0	is:1	one:0	second:1	the:1	third:0	this:1	

And this is the third one. [1 0 0 1 1 0 1 1 1]
and:1	document:0	first:0	is:1	one:1	second:0	the:1	third:1	this:1	

Is this the first document? [0 1 1 1 0 0 1 0 1]
and:0	document:1	first:1	is:1	one:0	second:0	the:1	third:0	this:1	

Test Vectors
This is the fourth document. [0 1 0 1 0 0 1 0 1]
and:0	document:1	first:0	is:1	one:0	second:0	the:1	third:0	this:1	

This document is the one. [0 1 0 1 1 0 1 0 1]
and:0	document:1	first:0	is:1	one:1	second:0	the:1	third:0	this:1	

This document is new. [0 1 0 1 0 0 0 0 1]
and:0	document:1	first:0	is:1	one:0	second:0	the:0	third:0	this:1
```

## TF\-IDF

문서를 TF-IDF 행렬로 변환

CountVectorizer → TfidfTransformer 적용한것과 같다

Fit learns **vocabulary** and **idf** from training set.

$$TF_{test}(w)\cdot \log{\frac{N_{train}}{DF_{train}(w)}}$$

- 참고 파라미터 (CountVectorizer 파라미터들도 동일하게 적용)
    - **norm: *{‘l1’, ‘l2’}, default=’l2’***
        - Each output row will have unit norm
        - 'l2': sum of **squares of vector elements** is 1 → 코사인 유사도=Dot Product 값 되도록
        - 'l1': sum of absolute values of vector elements is 1
    - **use_idf: *bool, default=True***
        - False일경우 IDF(t)=1로 계산
    - **smooth_idf: *bool, default=True***
        - DF에 1씩 더함 → 0 division 방지
    - **sublinear_tf: *bool, default=False***
        - sublinear tf scaling:
            - tf 대신 1+log(tf)

```python
This is the first document. 
[0.         0.46979139 0.58028582 0.38408524 0.         0.
 0.38408524 0.         0.38408524]
and:0.000	document:0.470	first:0.580	is:0.384	one:0.000	second:0.000	the:0.384	third:0.000	this:0.384	

This document is the second document. 
[0.         0.6876236  0.         0.28108867 0.         0.53864762
 0.28108867 0.         0.28108867]
and:0.000	document:0.688	first:0.000	is:0.281	one:0.000	second:0.539	the:0.281	third:0.000	this:0.281	

And this is the third one. 
[0.51184851 0.         0.         0.26710379 0.51184851 0.
 0.26710379 0.51184851 0.26710379]
and:0.512	document:0.000	first:0.000	is:0.267	one:0.512	second:0.000	the:0.267	third:0.512	this:0.267	

Is this the first document? 
[0.         0.46979139 0.58028582 0.38408524 0.         0.
 0.38408524 0.         0.38408524]
and:0.000	document:0.470	first:0.580	is:0.384	one:0.000	second:0.000	the:0.384	third:0.000	this:0.384	

Test TFIDF
This is the fourth document. 
[0.         0.57684669 0.         0.47160997 0.         0.
 0.47160997 0.         0.47160997]
and:0.000	document:0.577	first:0.000	is:0.472	one:0.000	second:0.000	the:0.472	third:0.000	this:0.472	

This document is the one. 
[0.         0.42796959 0.         0.34989318 0.67049706 0.
 0.34989318 0.         0.34989318]
and:0.000	document:0.428	first:0.000	is:0.350	one:0.670	second:0.000	the:0.350	third:0.000	this:0.350	

This document is new. 
[0.         0.65416415 0.         0.53482206 0.         0.
 0.         0.         0.53482206]
and:0.000	document:0.654	first:0.000	is:0.535	one:0.000	second:0.000	the:0.000	third:0.000	this:0.535
```

<!-- # LM Based -->