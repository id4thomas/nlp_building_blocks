# 2602-01-agents
agent 프레임워크, fastmcp 실험

## 구현 내용
- 위치 정보 (server_location) / 날씨 정보 (server_weather) tool을 제공하는 FastMCP 서버
- 위치 정보 & 날씨 정보를 종합적으로 활용하는 쿼리들을 처리하는 에이전트
    - ex. 현재 위치 정보를 조회 해서 주변 날씨를 검색

다음 시나리오들을 테스트한다


## 대상 프레임워크
아래 프레임워크들로 Agent 구현을 실험한다

| Framework | 핵심 | 상태 관리 |
| --- | --- | --- |
| [MS Agent](./ms-agent/) | 멀티에이전트 대화 | 오케스트레이터 컨텍스트 |
| AG2 | 멀티에이전트 대화 | 대화 히스토리 |
| LangGraph | 상태 그래프 | 명시적 State + 체크포인트 |


## MCP Server
[mcp_servers](./mcp_servers/) 폴더에 FastMCP Server 구현 내용을 자세히 기술한다
- server_location / server_weather 각각 별도의 MCP 서버로 구현



