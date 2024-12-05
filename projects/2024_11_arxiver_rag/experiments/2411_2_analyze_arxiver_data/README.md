# 2411_2_analyze_arxiver_data
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