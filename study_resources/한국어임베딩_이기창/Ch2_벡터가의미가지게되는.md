# Ch2. 벡터가 의미 가지게 되는



## 임베딩을 만드는 3가지 철학
임베딩에 자연어 의미를 함축: 자연어의 통계적 패턴(Statistical Pattern) 정보를 임베딩에 넣음

1. Bag-of-Words 가정: 어떤 단어가 **많이** 쓰였는가
    
2. Language Model (언어 모델): 단어가 **어떤 순서로** 쓰였는가
    
3. Distributional Hypothesis (분포 가정): 어떤 단어가 **같이** 쓰였는가
    


## 1. Bag-of-Words 가정

원소의 순서는 고려하지 않는다

1. BoW Embedding
    
    빈도 카운트값, 등장 여부(0,1) 만 사용하는 경우도 존재
    

2. TF-IDF: 빈도값 씀으로써 생기는 문제 해결
    
    **모든 도큐멘트에 많이** 나오는 단어에 집중 되는 **문제 해결**
    
    <!-- $$TF(W)\cdot\log{\frac{N}{DF(W)}}$$ -->
    <img src="https://render.githubusercontent.com/render/math?math=TF(W)\cdot\log{\frac{N}{DF(W)}}">
    
    * TF(W): **특정 문서에** 얼마나 많이 쓰엿는지
    
    * N: 전체 문서 수
    
    * DF(W): 단어가 **나타난** **문서 수**
    
    * log (N / DF(W)): IDF (log 까지 포함)
    
    <!-- ** 각 word의 TF-IDF는 **document 별로** 계산된다 -->
    
    Row: Term, Column: Document → 각 word TF-IDF 값을 **document 별로** 계산

3. Deep Averaging Network (Iyyer 15)
    
    문장 내 단어 임베딩 평균을 문장 임베딩으로
    
    이 문장 임베딩 입력으로 분류 태스크
    

## 2. Language Model

단어 시퀀스에 확률 (probability) 부여하는 방식

조건부 확률 (conditional probability) 이용해 최대 우도 추정법(Maximum Likelihood Estimation) 으로 유도

n-gram 모델으로 직전 n-1개 단어 등장 확률으로 전체 단어 시퀀스 등장확률 근사 (approximate)

→ Markov Assumption: 한 상태는 그 직전 상태에만 의존한다

- 존재하지 않는 단어가 확률 0이 되는 문제
    - 백오프 (Back-off): n-gram 등장 빈도를 **n 보다 작은 범위**의 시퀀스로 **근사**
        - alpha,beta 파라미터, n: 7→ 4 경우
        - Freq(7-gram) ~= alpha*Freq(4-gram) + beta
    - 스무딩 (Smoothing): 등장 빈도에 모두 K 만큼 더함 (add-k smoothing)
        - k=1 이면 laplace smoothing
        
- 뉴럴 네트워크 기반 모델
    - 다음 단어 예측 (unidirectional): ELMo, GPT
    - 마스크 언어 모델 (Masked LM): 문장 전체를 보고 중간(mask)를 예측. 양방향 학습 가능 (bi-directional) BERT

## 3. Distributional Hypothesis (분포 가정)

### **분포 (Distribution)**

특정 범위 (window) 내에 동시에 등장하는 문맥 (context), 이웃 단어의 집합

어떤 단어쌍이 비슷한 문맥 환경에 자주 등장 → 의미 또한 유사 할 것

ex) 빨래 & 세탁 같이 **주변에 등장하는 단어가 유사**

**분포 가정**: 비슷한 의미를 지닐 가능성 높다

But, 분포 정보와 의미 사이에 **직접적 연관성 보이지는 X**

### **분포와 의미**

1. 형태소 (morpheme): (어휘적/문법적) **의미**를 가지는 **최소 단위**
    
    형태소 분석 방법:
    
    계열 관계 (Paradigmatic Relation) - 해당 형태소 자이레 다른 형태소가 대치 되어 쓰일 수 있는가
    
    타깃 단어 주변 문맥 정보로 형태소를 확인
    
2. 품사: **문법적 성질** 따라 묶인 것
    
    분류 기준: 기능 (function), 의미 (meaning), 형태 (form)
    
    - 기능: 문장 내에서 다른 단어와 맺는 관계
        - 문장 내 역할 (주어, 서술어,..)
    - 의미: 단어의 형식적 의미 (사물의 이름을 나타내느냐, 성질/상태를 나타내느냐)
        - 어휘적 의미: (깊이, 깊다), (높이, 높다)
        - 형식적 의미: (깊이, 높이), (깊다, 높다)
    - 형태: 형태적 특징
        - 어미가 붙어 여러 모습으로 변화
    

의미 & 형태는 품사 분류 시 고려 대상이 될 수 있으나 결정적 기준이 될 수 없다

- 의미:
    - 공부하다 / 공부
    - '공부하다'는 동사 '공부'는 명사로 표현하지만 '공부'라는 단어의 **의미**에 움직임을 내포하는 **의미**가 없다고 할 수 없다.
    - 의미만으로 구분했다면 '공부' 명사라고 할 수 있을것인가
- 형태:
    - a) *영수*가 학교에 간다
    - b) *영수*! 조용히 해
    - 영수가 a에서는 명사, b에서는 감탄사 → 형태는 같지만 기능,의미가 다르다
    

**기능**과 **분포**는 개념적으로 엄밀히 다르지만 **밀접한 관련**을 지닌다

→ 임베딩에 **분포 정보**를 **함축**하게 되면 해당 벡터에 해당 **단어의 의미를** 자연스레 **내재**시킬 수 있게 된다

### **PMI (Pointwise Mutual Information) - 점별 상호 정보량**

두 확률 변수 (random variable) 사이 상관성을 계량화 하는 단위

독립일 경우 0

PMI: 두 단어의 등장이 **독립일 때 대비**해 얼마나 자주 등장 하는지

<!-- $$PMI(A;B) = \log{\frac{p(A,B)}{p(A)p(B)}}$$ -->
<img src="https://render.githubusercontent.com/render/math?math=PMI(A,B)=\log{\frac{p(A,B)}{p(A)p(B)}}">

PMI 행렬의 **행** 자체를 단어 임베딩으로 사용 가능

단어-문맥 행렬 (word-context matrix)

* ex) [개울가,에서,속옷*,**빨래***,를,하는,남녀]

    * window 2인 경우: **빨래** row에 대해 **에서,속옷,를,하는** column +=1

PMI 계산 예시:
* 전체 빈도: 1000
* 빨래: 20, 속옷: 15, 빨래 & 속옷: 10 인 경우

<!-- $$\log{\frac{\frac{10}{1000}}{\frac{20}{1000}\frac{15}{1000}}}$$ -->
<img src="https://render.githubusercontent.com/render/math?math=\log{\frac{\frac{10}{1000}}{\frac{20}{1000}\frac{15}{1000}}}">

### **Word2Vec**

Word2vec 기법이 PMI와 연관이 깊다

- CBOW: 문맥 단어 가지고 타깃 단어를 예측
- Skip-Gram: 타깃 단어로 문맥 단어들을 예측