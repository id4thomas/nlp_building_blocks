# 2602-01-agents
agent 프레임워크, fastmcp, tracing 실험

MCP/ A2A / AG-UI 프로토콜에 대한 이해도 높이는 것이 목적
- 파악 Medium 블로그 게시글: [Agent 프로토콜 파악 (2025-02)](https://medium.com/@id2thomas/agent-%ED%94%84%EB%A1%9C%ED%86%A0%EC%BD%9C-%ED%8C%8C%EC%95%85-2026-02-f6a763fd5ea3)

## 구현 내용
- 위치 정보 (server_location) / 날씨 정보 (server_weather) tool을 제공하는 FastMCP 서버
- 위치 정보 & 날씨 정보를 종합적으로 활용하는 쿼리들을 처리하는 에이전트
    - ex. 현재 위치 정보를 조회 해서 주변 날씨를 검색
- tracing 프레임워크를 통한 실행 시각화


## MCP Server
[mcp_servers](./mcp_servers/) 폴더에 FastMCP Server 구현 내용을 자세히 기술한다
- server_location / server_weather 각각 별도의 MCP 서버로 구현

## Agent 프레임워크
아래 프레임워크들로 Agent 구현을 실험한다

후보군:
| 실험 폴더 | 핵심 | 상태 관리 | 공식 링크 |
| --- | --- | --- | --- |
| [MS Agent](./agents/ms-agent/) | 멀티에이전트 대화 | 오케스트레이터 컨텍스트 | [링크](https://learn.microsoft.com/en-us/agent-framework/overview/?pivots=programming-language-python) |
| [Langchain/LangGraph](./agents/langchain/) | 상태 그래프 | 명시적 State + 체크포인트 | [링크](https://www.langchain.com/agents) |
| [PydanticAI](./agents/pydanticai/) | | | [링크](https://ai.pydantic.dev) |
| [Google ADK](./agents/google-adk/) | | | [링크](https://google.github.io/adk-docs/) |
| [OpenAI Agents](./agents/openai-agents/) | | | [링크](https://openai.github.io/openai-agents-python/) |
| AG2 | 멀티에이전트 대화 | 대화 히스토리 | [링크](https://www.ag2.ai) |

각 프레임워크 별로 아래 내용을 파악한다
- LLM Client 선언
- (Remote) MCP Tool 연결
- Agent 선언, 실행 (+스트리밍)
- 실행 결과

## Tracing / Observability