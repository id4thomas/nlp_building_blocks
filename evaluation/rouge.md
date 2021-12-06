# ROUGE
Analyzing ROUGE
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is recall oriented measure for evaluating text generation systems<br>
Usually used for evaluating <b>text summarization & machine translation</b><br>
Compares model generated sentence with reference sentence.

## Variations
* ROUGE-N (ROUGE-1, ROUGE-2,..)
    * Counts overlapping n-grams
* ROUGE-L
    * Uses <b>LCS (Longest Common Subsequence)</b> to measure longest matching subsequence.
    * Doesn't require continuous matching (연속적 매칭 요구하지 않음)
        * 어떻게든 발생하는 매칭 측정 -> 유연한 성능 비교 위해
* ROUGE-S (Skip-gram Co-ocurrence)
    * Calculate overlap but use <b>Skip-grams</b>
    * When given <i>Window Size</i> 2
        * Example: "This is example sentence"
        * "This is", "This example", "This sentence", "is example", "is sentence", "example sentence"

## Reporting Scores
* ROUGE metric score returns <b>Precision</b> and <b>Recall</b> values.
    * Recall 값이 좋더라도 Summarization 결과가 길다면 관련 없는 단어들이 들어가면서 어떻게던 정답 단어들이 들어갈 수 있음
    * 결과에 불필요한 단어들이 많다면 Precision 값이 좋지 않을것
    * Best to report F-measure values, but if sentences are short Recall values are acceptable.

### Multiple References
How ROUGE handles multiple references.
<!-- For single prediction, take <b>Maximum</b> ROUGE score <b>among all references</b>. (ROUGE-N, ROUGE-L)
* ROUGE-N(prediction,references) = max_k(ROUGE-N(prediction,reference_k)) -->

From the following [issue](https://github.com/pltrdy/rouge/issues/17)
```
I never used it myself, but I know the original ROUGE (aka. ROUGE-1.5.5) handles multiple references by either averaging scores accross references (default) or using the best score. It's controlled by the f flag, see https://github.com/pltrdy/files2rouge/blob/
```

```
Pre-1.5 version of ROUGE used model average to compute the overall
    ROUGE scores when there are multiple references. Starting from v1.5+,
    ROUGE provides an option to use the best matching score among the
    references as the final score. The model average option is specified
    using "-f A" (for Average) and the best model option is specified
    using "-f B" (for the Best). The "-f A" option is better when use
    ROUGE in summarization evaluations; while "-f B" option is better when
    use ROUGE in machine translation (MT) and definition
    question-answering (DQA) evaluations since in a typical MT or DQA
    evaluation scenario matching a single reference translation or
    definition answer is sufficient. However, it is very likely that
    multiple different but equally good summaries exist in summarization
    evaluation.
```
* By default "scoring formula": compute <b>average</b>
* <b>Average</b> is better when using ROUGE in <b>summarization</b> evaluations
* <b>Best</b> is better when using ROUGE in Machine Translation and Definition Question-Answering
    * Matching a single reference is sufficient


## Averaging
From <i>_get_avg_scores</i> function at https://github.com/pltrdy/rouge/blob/master/rouge/rouge.py<br>
Get scores for each hypothesis-reference pair and take <b>arithmetic mean</b>
```python
count = 0
for (hyp, ref) in zip(hyps, refs):
    ...

    for m in self.metrics:
        fn = Rouge.AVAILABLE_METRICS[m]
        sc = fn(hyp, ref, exclusive=self.exclusive)
        scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}
    ...
    count += 1
avg_scores = {
    m: {s: scores[m][s] / count for s in self.stats}
    for m in self.metrics
}
```

# ROUGE Implementations
* [files2rouge](https://github.com/pltrdy/files2rouge) (Official Rouge) 
    * Also provides python implementation ([rouge](https://github.com/pltrdy/rouge) pip install rouge)
* [rouge-metric](https://github.com/li-plus/rouge-metric)
    * A Python wrapper of the official ROUGE-1.5.5.pl
    * (pip install rouge-metric)
* Huggingface datasets metric [[link](https://huggingface.co/metrics/rouge)]

Examples for each package at [rouge.ipynb](./rouge.ipynb)

## Step-by-Step
Analyzing ROUGE-L implementation of <i>rouge-metric</i> package.
```python
hypotheses = [
      doc1_hyp_summary,   # Hypothesis summary for document 1
      doc2_hyp_summary,   # Hypothesis summary for document 2
      ...
  ]
multi_references = [
    [
        doc1_ref1_summary,  # Reference summary 1 for document 1
        doc1_ref2_summary,  # Reference summary 2 for document 1
        ...
    ],
    [
        doc2_ref1_summary,  # Reference summary 1 for document 2
        doc2_ref2_summary,  # Reference summary 2 for document 2
        ...
    ],
]
```
1. Split into sentences and tokenize each sentence 
    * evaluate()
```python
#In evaluate() method of PyRouge.
tokenized_hyp = [[tokenizer(sent) for sent in sentencizer(hyp)] for hyp in hypotheses]
tokenized_multi_ref = [[[tokenizer(sent) for sent in sentencizer(ref)] for ref in multi_ref] for multi_ref in multi_references]
```
2. Call score function for each (hypothesis,<b>references</b>) pair.
```python
#In evaluate_tokenized() method of PyRouge
result = aggregator.aggregate(
        _rouge_scores_multi_ref(
            hyp, multi_ref, self.rouge_n, self.rouge_l, self.rouge_w,
            self.rouge_w_weight, self.rouge_s, self.rouge_su, self.skip_gap,
            self.multi_ref_mode, self.alpha
        ) for hyp, multi_ref in zip(hypotheses, multi_references)
    )
```
3. Call score function for each (hypothesis,<b>reference</b>) pair.
```python
#_rouge_scores_multi_ref function
agg = _build_match_aggregator(multi_ref_mode)
    
#Calculate with each hyp-ref pair & aggregate
match = agg.aggregate(
    _rouge_l_summary_level(hyps, refs) for refs in multi_refs)
```

4. Calculate Rouge-L score
```python
# _rouge_l_summary_level function
def _rouge_l_summary_level(hyps, refs):
    # list of hyp, ref tokens
    # type: (List[List[str]], List[List[str]]) -> _Match
    hyp_unigram = _build_ngrams(_flatten(hyps), 1)
    match_size = 0
    for ref in refs:
        lcs_union = _lcs_union(hyps, ref)
        for ref_idx in lcs_union:
            unigram = (ref[ref_idx],)
            if hyp_unigram.get(unigram, 0) > 0:
                hyp_unigram[unigram] -= 1
                match_size += 1
    ref_len = sum(len(ref) for ref in refs)
    hyp_len = sum(len(hyp) for hyp in hyps)
    return _Match(match_size, hyp_len, ref_len)
```

## ETC
* Different scores across implementations? 
    * https://github.com/google/seq2seq/issues/89
    * Need to consider Stemming, Tokenization
``` 
The "official" ROUGE script does a bunch of stemming, tokenization, and other things before calculating the score. 
The ROUGE metric in here doesn't do any of this, but it's a good enough proxy to use during training for getting a sense of what the score will be. 
As the amount of data increases and sentences become more similar it should be relatively close (at least in my experiments)

So the recommended thing to do is to still run the official ROUGE script on the final model if you want to compare to published results.

I don't want to use pyrouge, or some kind of other wrapper around the ROUGE script, because it's

A real pain to install and get working on various machines
Not openly available, at least not "officially"
```


## References
[1] https://huffon.github.io/2019/12/07/rouge/<br>
[2] https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460<br>
[3] https://github.com/pltrdy/rouge<br>
[4] https://github.com/li-plus/rouge-metric<br>
[5] https://github.com/pltrdy/rouge/issues/2<br>
[6] https://github.com/google/seq2seq/issues/89<br>
[7] https://www.mathworks.com/help/textanalytics/ref/rougeevaluationscore.html<br>
[8] ROUGE: A Package for Automatic Evaluation of Summaries - Chin-Yew Lin