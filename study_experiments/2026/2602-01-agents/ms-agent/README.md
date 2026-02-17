# Microsoft Agent Framework

## 프레임워크 파악
MS AutoGen, semantic-kernel의 후속 프레임워크
- [MS 공식 비교 블로그 (2025.10.07)](https://devblogs.microsoft.com/semantic-kernel/semantic-kernel-and-microsoft-agent-framework/)


### Client vs Agent?
client가 LLM 백엔드 정보, Agent는 페르소나 (인스트럭션)을 부여 받은 객체 (client를 가짐)
- client가 에이전트 팩토리 느낌?

## 구현 기초
### Client 선언
Client: 사용할 LLM Provider에 해당되는 Client를 선언
- 다 core._clients의 BaseChatClient를 상속 (소스 [링크](https://github.com/microsoft/agent-framework/blob/54a67d96cd297c7d49ff7de9c8641534034a5468/python/packages/core/agent_framework/_clients.py#L197C7-L197C21))
- client의 `as_agent` 메서드를 통해 Agent를 생성

OpenAI compatible의 경우 OpenAIChatClient (chat-completions API), OpenAIResponsesClient (responses API) 제공
- hosted MCP를 사용하려면 OpenAIResponsesClient를 써야 함 (chat쪽은 지원 x)

예시:
- 로컬 vLLM 서버 연결
- OpenAI 클라이언트를 직접 선언해서 주는 것도 가능
```
from agent_framework.openai import OpenAIResponsesClient

## 필요 파라미터 넘겨서 선언하는 경우

client = OpenAIResponsesClient(
    base_url="http://localhost:9001/v1",
    api_key="sk-123",
    model_id="Qwen3-VL-30B-A3B-Instruct"
)

## OpenAI Client를 직접선언해서 넘기는 경우
client = OpenAIResponsesClient(
    async_client = AsyncOpenAI(
        base_url="http://localhost:9001/v1",
        api_key="sk-123",
    ),
    model_id="Qwen3-VL-30B-A3B-Instruct"
)
```

### MCP Tool 연결
공식 문서 기준 mcp tool 조회는 방법은 2가지
- 방식1: client의 `get_mcp_tool` 메서드를 사용해서 조회
- 방식2: 직접 MCP*Tool 클래스로 선언
방식2 같이 명시적으로 선언해서 넘기는 것이 좋아보임

**[방식 1]**

[가이드](https://learn.microsoft.com/en-us/agent-framework/agents/providers/openai?pivots=programming-language-python)의 "Hosted Tools with Responses Client" 섹션 기준

```
client = OpenAIResponsesClient()

## OpenAI 공식 tool
code_interpreter = client.get_code_interpreter_tool()
web_search = client.get_web_search_tool()
file_search = client.get_file_search_tool(vector_store_ids=["vs_abc123"])

## 직접 선언한 mcp tool
mcp_tool = client.get_mcp_tool(
    name="location",
    url="http://localhost:8001/mcp/",
    approval_mode="never_require",
)
```

mcp_tool은 아래와 같이 딕셔너리로 반환됨
```
{'type': 'mcp',
 'server_label': 'location',
 'server_url': 'http://localhost:8001/mcp/',
 'require_approval': 'never'}
```


이슈:
- 해당 방식으로 tool 조회해서 넘길시 Agent가 tool을 제대로 못찾는 것으로 보임
- OpenAIResponsesClient쪽 [소스](https://github.com/microsoft/agent-framework/blob/54a67d96cd297c7d49ff7de9c8641534034a5468/python/packages/core/agent_framework/openai/_responses_client.py#L622)


**[방식 2]**

mcp transport에 따라서 아래 mcp tool 클래스를 제공
- MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
- 직접 httpx 클라이언트를 제공 가능
    - 직접 headers 정보 추가 가능
    - follow_redirects=True 세팅 필수 (참고: mcp python-sdk [소스](https://github.com/modelcontextprotocol/python-sdk/blob/705497a59369eec487b04c82672d4ea60e795298/src/mcp/shared/_httpx_utils.py#L69))

http-streamable (hosted mcp) 기준으로만 우선 파악

```
from agent_framework import MCPStreamableHTTPTool

location_tools = MCPStreamableHTTPTool(
    name="location",
    url="http://localhost:8001/mcp/",
    approval_mode="never_require",
    http_client = httpx.AsyncClient(
        headers={"x-user-id": "1234"},
        timeout=60.0,
        follow_redirects=True
    )
)

weather_tools = MCPStreamableHTTPTool(
    name="weather",
    url="http://localhost:8002/mcp/",
    approval_mode="never_require",
)
```

`load_tools`메서드로 tool 디스커버리 실행시
```
# load_tools=True일 시 아래 load_tools 메서드 중복으로 실행됨 (디버깅때만 세팅)
weather_tools = MCPStreamableHTTPTool(
    name="weather",
    url="http://localhost:8002/mcp/",
    approval_mode="never_require",
    load_tools=False
)

# mcp 서버에서 가능 tool 조회
await weather_tools.load_tools()

function_names = [x.name for x in weather_tools.functions]
print(f"{len(function_names)} tools - {function_names}")

# 2 tools - ['get_current_weather', 'get_weather_forecast']
```

## Agent 선언, 실행
Client.as_agent로 에이전트 선언

```
agent = client.as_agent(
    name="HelpfulAssistant",
    instructions=AGENT_INSTRUCTION,
    tools=[location_tools, weather_tools],
    conversation_id=None
)
```

유저 쿼리를 아래같이 실행
- result는 AgentResponse 객체
```
# options 파라미터는 OpenAI responses create 메서드까지 넘겨짐
# vLLM 서버의 경우 store 지원 x, True 일시 (기본값) 'response id {} not found' 404 오류 발생
result = await agent.run(
    query,
    session=None,
    options={"store": False}
)
```

### 에이전트 Response
agent.run 실행시 AgentResponse 객체를 반환
- [소스](https://github.com/microsoft/agent-framework/blob/54a67d96cd297c7d49ff7de9c8641534034a5468/python/packages/core/agent_framework/_types.py#L2112)

Response내 정보 계층
- AgentResponse가 self.messages로 List[Message]를 가짐
- Message 각각 List[Content]를 가짐
- Content는 function_call, function_result 같은 정보를 가짐


## Single Agent 실행 예시
쿼리: "내 현재 위치 주변 도시들을 알려주고 그 도시들의 날씨들도 알려줘. 도시 명칭은 반환값 그대로 사용해"

생성 답변 (print(results))
```
현재 위치인 부산 주변 도시로는 김해, 창원, 울산이 있습니다. 각 도시의 날씨는 다음과 같습니다:

- **김해**: 구름 조금, 기온 8°C, 습도 60%, 바람 19km/h  
- **창원**: 날씨 정보를 확인 중입니다.  
- **울산**: 날씨 정보를 확인 중입니다.  

추가로 창원과 울산의 날씨도 확인해드릴까요?
```

### result에 담긴 messages 내용 (실행 순서)
확인 코드:
```
for i, message in enumerate(result.messages):
    contents = message.contents
    print(f"MSG {i} {message.role} - {len(contents)} contents")
    for cont_i, content in enumerate(contents):
        print(f"CONT {cont_i} - {content.type}")
        if content.type=='text':
            print(f"\t{repr(str(content))}")
        elif content.type=='function_result':
            print(f"\t{content.name}\t{content.result}")
        else:
            print(f"\t{content.name}\t{content.arguments}")
    print('-'*30)
```

내부 내용:
- assistant-tool-assistant-.. 순서로 tool calling이 실행됨
- assistant: function_call을 생성 -> tool: function_result를 받아옴
```
MSG 0 assistant - 1 contents
CONT 0 - function_call
	get_user_location	{}
------------------------------
MSG 1 tool - 1 contents
CONT 0 - function_result
	None	{"name":"Busan","coordinate":{"lat":35.1796,"lon":129.0756}}
------------------------------
MSG 2 assistant - 1 contents
CONT 0 - function_call
	get_nearby_cities	{"city_name": "Busan"}
------------------------------
MSG 3 tool - 1 contents
CONT 0 - function_result
	None	[{"name":"Gimhae","coordinate":{"lat":35.2285,"lon":128.8894}},{"name":"Changwon","coordinate":{"lat":35.228,"lon":128.6811}},{"name":"Ulsan","coordinate":{"lat":35.5384,"lon":129.3114}}]
------------------------------
MSG 4 assistant - 4 contents
CONT 0 - function_call
	get_current_weather	{"city_name": "Busan"}
CONT 1 - function_call
	get_current_weather	{"city_name": "Gimhae"}
CONT 2 - function_call
	get_current_weather	{"city_name": "Changwon"}
CONT 3 - function_call
	get_current_weather	{"city_name": "Ulsan"}
------------------------------
MSG 5 tool - 4 contents
CONT 0 - function_result
	None	{"time":"20260217-22:00","status":{"temp_cel":6,"condition":"구름 조금","humidity_pct":59,"wind_kph":19}}
CONT 1 - function_result
	None	{"time":"20260217-22:00","status":{"temp_cel":8,"condition":"맑음","humidity_pct":56,"wind_kph":12}}
CONT 2 - function_result
	None	{"time":"20260217-22:00","status":{"temp_cel":7,"condition":"구름 조금","humidity_pct":58,"wind_kph":11}}
CONT 3 - function_result
	None	{"time":"20260217-22:00","status":{"temp_cel":7,"condition":"흐림","humidity_pct":61,"wind_kph":17}}
------------------------------
MSG 6 assistant - 1 contents
CONT 0 - text
	'id4thomas님, 현재 위치인 부산과 주변 도시의 날씨 정보를 알려드릴게요.\n\n- **부산**: 6°C, 구름 조금, 습도 59%, 풍속 19km/h  \n- **김해**: 8°C, 맑음, 습도 56%, 풍속 12km/h  \n- **창원**: 7°C, 구름 조금, 습도 58%, 풍속 11km/h  \n- **울산**: 7°C, 흐림, 습도 61%, 풍속 17km/h  \n\n날씨 참고해 주세요!'
------------------------------
```


## References
Agent Framework 공식 가이드
- [Provider - OpenAI Agents](https://learn.microsoft.com/en-us/agent-framework/agents/providers/openai?pivots=programming-language-python)
- [Using MCP tools with Agents](https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools?pivots=programming-language-python)
- [Running Agents](https://learn.microsoft.com/en-us/agent-framework/agents/running-agents?pivots=programming-language-python)