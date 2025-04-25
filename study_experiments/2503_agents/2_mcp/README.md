# MCP
test mcp protocol

## 실행 방식
1. stdio 방식
* 호스트에서 직접 실행

등록 예시:
* 로컬 환경에서 실행 path 지정 필요
```
{
    "mcpServers": {
        "weather": {
            "command": "/opt/miniconda3/envs/py310/bin/uv",
            "args": [
                "--directory",
                "/Users/id4thomas/github/nlp_building_blocks/study_experiments/2503_agents/2_mcp/mcp-server-demo",
                "run",
                "main.py"
            ]
        }
    }
}
```

2. sse 방식 (server side event)
* 웹서버를 실행하고 연결하는 방식

등록 예시:
```
{
    "mcpServers": {
        "weather": {
            "url": "http://localhost:8000/sse"
        }
    }
}
```

## Resources
* https://rudaks.tistory.com/entry/MCP-Server-개발-Python
* https://hyeong9647.tistory.com/entry/MCP-Remote-SSE-사용하기