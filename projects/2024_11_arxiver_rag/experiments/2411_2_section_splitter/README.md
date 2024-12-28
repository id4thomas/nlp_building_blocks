# 2411_2_section_splitter
* 데이터를 헤더 (섹션) 단위로 분리하는 모듈을 개발한다

## Experiments
* 1_read_arxiver_data
    * arxiver parquet 데이터 읽고 길이 분석 (abstract, sections)
* 2_test_markdown_splitter
    * Markdown 형식의 텍스트를 분할하기 위한 로직 개발, 샘플 확인
    * Package options
        * langchain_text_splitters.MarkdownHeaderTextSplitter - doesn't keep hierarchy
        * md2py - md->html->bs4
            * https://github.com/alvinwan/md2py
        * mrkdwn_analysis - identifies components
            * https://github.com/yannbanas/mrkdwn_analysis
        * Markdown Tree
            * https://github.com/fmelon/MarkdownTree
* 3_test_section_splitter
    * Paper, Section 이라는 데이터 클래스를 만들어서 제대로 분할 진행
* 4_test_section_splitter_module
    * `sample.parquet` 샘플들로 처리 가능여부 테스트
    * 8 프로세스로 처리
```
(llm) id4thomas@YRMB14-2 2411_2_analyze_arxiver_data % python 4_test_section_splitter_module.py
(10000, 7) Index(['id', 'title', 'abstract', 'authors', 'published_date', 'link',
       'markdown'],
      dtype='object')
2677 RECURSION ERROR maximum recursion depth exceeded
Total 10000 failed 1
Elapsed 149.208
```

* 5_debug_md2py_max_recursion
    * `maximum recursion depth exceeded` 오류 샘플 디버깅
    * https://www.notion.so/241228-splitter-maximum-recusrion-depth-fdfc4755cf684f3bb394bd743bc5fce4?pvs=4

* 6_test_section_data
    * ArxivPaperSection 데이터 클래스 구성, 테스트 (3번 실험의 연장선)