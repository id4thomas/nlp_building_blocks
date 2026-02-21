# OpenAI Agents SDK
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

## 프레임워크 파악


## 구현 기초
### Client (Model) 선언
OpenAI 클라이언트를 `OpenAIChatCompletionsModel`/`OpenAIResponsesModel`에 제공해서 선언

예시:
```
llm_client=AsyncOpenAI(
    base_url="http://.../v1",
    api_key="sk-123",
)
model = OpenAIChatCompletionsModel(
    model="Qwen3-...",
    openai_client=llm_client,
)
```

### Tool 연결
`MCPServerManager`를 통해 원격 MCP 서버를 관리
- 공식 가이드 [[링크]](https://openai.github.io/openai-agents-python/mcp/)
- `MCPServerStreamableHttp`로 연결 설정을 정의

```
server1 = MCPServerStreamableHttp(
    name="location",
    params={
        "url": "http://.../mcp",
        "headers": {"x-user-id": user_id},
        "timeout": 60
    },
    cache_tools_list=True,
)
...

# Initialize Manager
servers = [server1, ...]
manager = MCPServerManager(servers)

async with manager:
    agent = Agent(..., mcp_servers=manager.active_servers)
```

### Agent 선언
```
agent = Agent(
    name="WeatherAgent",
    instructions=agent_instruction,
    model=model,
    mcp_servers=mcp_manager.active_servers,
    model_settings=ModelSettings(
        parallel_tool_calls=True,
        temperature=0.8,
        ...
    )
)
```

## Agent 실행 예시
Agent를 실행할때는 `Runner`를 사용해야 함
- `RunResult` 객체를 반환함 ([공식 문서](https://openai.github.io/openai-agents-js/openai/agents/classes/runresult/))

### run
```
from agents import Runner

result = await Runner.run(agent, input=query)
```

결과 확인
- new_items: 해당 run으로 생성된 아이템들
- final_output: 최종 생성 결과
```
for item in result.new_items:
    print(item.type)
    if item.type == "tool_call_item":
        raw = item.raw_item
        # MCP 툴인 경우
        tool_name = getattr(raw, "name", None) or getattr(raw, "tool_name", None)
        arguments = getattr(raw, "arguments", None)
        print(f"[tool_call] {tool_name}  args={arguments}")

    elif item.type == "tool_call_output_item":
        print(f"[tool_result] {item.output}")

    elif item.type == "message_output_item":
        print(f"[message] {ItemHelpers.text_message_output(item)}")

print(f"\n[final] {result.final_output}")
```

### stream
```
from openai.types.responses import ResponseTextDeltaEvent

current_text = ""
in_tool_call = False

result = Runner.run_streamed(agent, input=query)

async for event in result.stream_events():
    if event.type == "raw_response_event":
        ...
    elif event.type == "run_item_stream_event":
        item = event.item
        if item.type == "tool_call_item":
            # tool call 실행 요청
            ...
        elif item.type == "tool_call_output_item":
            # tool 호출 결과
            ...
        elif item.type == "message_output_item":
            # 텍스트 스트림이 끝났음을 표시
            ...
    elif event.type == "agent_updated_stream_event":
        # agent 가 할당됨
        print(f"\n🤖 [agent] {event.new_agent.name}\n")
```