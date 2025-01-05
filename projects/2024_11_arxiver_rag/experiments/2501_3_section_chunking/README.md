# 2412_1_section_chunking
* semantic-chunking 기반의 섹션 청킹기 개발

## Experiments
### 1_test_chunking_formatting
* sample 100개 정도 처리 후 청크 구성 테스트
* semantic_chunkers 패키지의 `StatisticalChunker`로 청킹
    * https://www.aurelio.ai/learn/semantic-chunkers-intro
    * 유사도 비교 모델로 baai/bge-m3 사용 (text-embedding-inference로 띄워둠)
* 청킹 후 짧은 청크(<128 토큰)는 앞 텍스트로 머징
* 섹션 제목 활용해서 포매팅

포매팅 템플릿
* p_chunk_template: Subsection이 'p' header일 경우
* h3_chunk_template: Subsection이 'h3' header일 경우
```
p_chunk_template = '''Section Title: "{section_title}"
Text:
{text}'''

h3_chunk_template = '''Section Title: "{section_title}"
Subsection Title: "{subsection_title}"
Text:
{text}'''
```