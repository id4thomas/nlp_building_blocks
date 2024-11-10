# 2411_2_analyze_arxiver_data
* 1_read_arxiver_data
    * arxiver parquet 데이터 읽고 길이 분석 (abstract, sections)
* 2_test_markdown_splitter
    * Package options
        * langchain_text_splitters.MarkdownHeaderTextSplitter - doesn't keep hierarchy
        * md2py - md->html->bs4
            * https://github.com/alvinwan/md2py
        * mrkdwn_analysis - identifies components
            * https://github.com/yannbanas/mrkdwn_analysis