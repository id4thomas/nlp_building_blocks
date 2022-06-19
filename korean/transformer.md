# Transformer Models

Publicly available pretrained korean models

## Encoder Models
* BERT
    * SKT/KoBERT
        * https://github.com/SKTBrain/KoBERT
        * huggingface: “skt/kobert-base-v1”
        * Commercial 라이선스
    * monologg/kobert
        * https://github.com/monologg/KoBERT-Transformers
        * huggingface: "monologg/kobert”
        * Commercial 라이선스
    * klue/bert-base
        * huggingface: “klue/bert-base”
        * Commercial 라이선스 (CC-BY-SA-4.0 license)
    * ETRI - KorBERT
        * https://aiopen.etri.re.kr/service_dataset.php
* RoBERTa
    * klue/roberta
    * huggingface: “klue/roberta-base” (-small, -large도 제공)
    * https://huggingface.co/klue/roberta-base
    * Commercial 라이선스 (CC-BY-SA-4.0 license)
* DistillBERT
    * monologg/distillkobert
    * huggingface: “monologg/distillkobert”
    * Commercial 라이선스 (Apache License 2.0)
* ELECTRA
    * monologg/koelectra
    * https://github.com/monologg/KoELECTRA
    * huggingface “monologg/koelectra-base-v3-discriminator”
        * 종류가 많으니 https://huggingface.co/monologg 참고
        * 분류 태스크에는 주로 discriminator 사용
    * Commercial 라이선스 (Apache-2.0)
* BigBird
    * monologg/kobigbird-bert-base
    * https://github.com/monologg/KoBigBird
        * BERT 토크나이저 사용
    * huggingface: “monologg/kobigbird-bert-base”

## Decoder Models
* GPT2
    * SKT/KoGPT2
    * huggingface: “skt/kogpt2-base-v2”
    * Non-commercial 라이선스
* GPT3
    * kakaobrain/KoGPT
    * huggingface: “kakaobrain/kogpt”
        * revision: “KoGPT6B-ryan1.5b”, “KoGPT6B-ryan1.5b-float16”
    * 소스코드는 Apache 2.0, 모델 웨이트는 Non-commercial (CC-BY-NC-ND 4.0)
* SKT - KoGPT Trinity
    * huggingface: “skt/ko-gpt-trinity-1.2B-v0.5”

## Encoder-Decoder Models
* T5
    * 전자기술연구원 AIRC-KETI/KE-T5
        * https://github.com/AIRC-KETI/ke-t5
        * Huggingface 모델, TF 모델 모두 제공
        * huggingface: “KETI-AIR/ke-t5-base” (버전 종류 많으니 깃헙 참고)
        * Commercial 라이선스 (Apache License 2.0)
    * ETRI - ET5
        * https://aiopen.etri.re.kr/service_dataset.php
* BART
    * SKT - KoBART
    * https://github.com/SKT-AI/KoBART
    * Commercial 라이선스 (Modified MIT License)
    * Transformers 포팅 버전
        * https://github.com/hyunwoongko/kobart-transformers
    * huggingface: “hyunwoongko/kobart”

## 세부 태스크 따른 모델
### SentenceBERT - 문장 임베딩
* KoSentenceBERT-SKT
    * https://github.com/BM-K/KoSentenceBERT-SKT

### NER 태스크
* eagle705 - BERT + CRF
    * https://github.com/eagle705/pytorch-bert-crf-ner