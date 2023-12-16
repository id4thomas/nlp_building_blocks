# BLEU
Bilingual Evaluation Understudy<br>
* How similar the candiate text is to the reference text, initially proposed to evaluate <b>Machine Translation</b>.
* Compute n-gram precision with <i>overlap</i> and <i>count</i>.<br>
* Independent of language, fast to calculate

## Methodology
Take geometric mean of test corpus' modified precision scores and multiply by an exponential brevity penalty factor.
* geometric average of modified n-gram precisions (n-grams up to length N)
    * Averaging weights used to perform weighted sum for each corresponding n-gram order. (Order weights)
    * Example:
        * BLEU-1: (1.0, 0.0, 0.0, 0.0)
        * BLEU-2: (0.5, 0.5, 0.0, 0.0)
        * BLEU-3: (0.33, 0.33, 0.33, 0.0)
        * BLEU-4: (0.25, 0.25, 0.25, 0.25)
            * BLEU-4: 0.25 * unigram score + 0.25 * bigram score + ..

### Modified n-gram precision
* Issue: MT system can <b>overgenerate</b> "reasonable" words
* Reference word should be considered <b>exhausted</b> after matching candidate word is identified
* Method:
    * First count maximum number of times a word occurs in <b>any single reference</b>
    * Clip to total count of each candidate word by its maximum reference count
        * min(Count(in Prediction), Max_ref_count)
    * Add clipped counts of each word and divide by total unclipped number of candidate words
* Example:
    * Candidate: the the the the the the the (7 'the's)
    * Ref1: <b>The</b> cat is on <b>the</b> mat
    * Ref2: There is cat is on <b>the</b> mat
    * Maximum count of "the" in references: 2
    * Modified Unigram precision = 2/7

### Brevity Penalty
Candidate translations longer than their references are already penalized by the <b>modified n-gram measure</b>.

Brevity Penalty Factor (BP): 
* high-scoring candidate must now match the reference translations in length, word choice, and in word order
* Use <b>best match length</b>
    * If prediction length is 12 and reference lengths are [12,15,17] -> brevity penalty=1
    * Call closest reference sentence length the "best match length"
* Compute <b>brevity penalty over the entire corpus</b> to allow some freedom at sentence level

BP Calculation:
* c: length of candidate
* r: test corpus' effective reference length
    * summing best match lengths for each candidate sentence

* BP = 1 if c > r
* BP = e^(1-r/c) if c <= r
    * decaying exponential in r/c
    * c가 짧을 수록 BP 값이 작아짐 (최대 1) -> <b>짧으면 Penalty 적용</b>

### Smoothing
When no ngram overlaps are found a smoothing function can be used.

## Issues with BLEU


<!-- ## Reporting Scores -->
## Multiple References
How bleu handles multiple references.
NLTK <i>corpus_bleu</i> function. [[Link]](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
```
Instead of averaging the sentence level BLEU scores (i.e. macro-average precision), the original BLEU metric (Papineni et al. 2002) accounts for the micro-average precision (i.e. summing the numerators and denominators for each hypothesis-reference(s) pairs before the division).
```
* Micro-average precision for single prediction with multiple refeerences.
* Average of sentence_bleu scores != corpus_bleu score

NLTK Implementation (<i>modified_precision</i> function)
```python
def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)
```

* counts: hypothesis n-gram counts
* referece_counts: n-gram count for that reference
* max_counts를 최종 intersection으로 이용
    * 'prediction'의 각 n-gram 마다 reference중 제일 많이 겹친 값으로 이용
    * -> 각 prected n-gram 마다 제일 좋게 봐주는 reference 기준 쓴다고 생각
* 최종 Prediction에 있는 해당 n-gram 개수로 clipping


## References
[1] https://jrc-park.tistory.com/273<br>
[2] https://machinelearningmastery.com/calculate-bleu-score-for-text-python/<br>
[3] https://wikidocs.net/31695<br>
[4] https://jrc-park.tistory.com/273<br>
[5] http://ssli.ee.washington.edu/~mhwang/pub/loan/bleu.pdf<br>
[6] https://www.nltk.org/_modules/nltk/translate/bleu_score.html<br>
[7] BLEU: a Method for Automatic Evaluation of Machine Translation - Papineni et al.