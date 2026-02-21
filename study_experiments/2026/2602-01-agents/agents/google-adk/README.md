# Google ADK
Google [ADK](https://google.github.io/adk-docs/) (Agent Development Kit)

## 프레임워크 파악


## 구현 기초
### Client (Model) 선언
OpenAI compatible 모델의 경우 adk 제공의 Litellm 기능을 사용해야 함
- Gemini / Anthropic은 별도 지원 클래스가 존재 :(

```
model = LiteLlm(
    model=f"openai/{model_name}", # openai compatible
    base_url="http://.../v1",
    api_key="sk-123",
)
```

### Tool 연결
toolset을 정의해서 Agent에 제공

```
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams

toolset1 = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://.../mcp",
        headers={cls._header_user_key: user_id},
        timeout=60,
    )
)

agent = LlmAgent(..., tools=[toolset1, ...])
```

### Agent 선언
```
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name=name,
    model=model,
    instruction=agent_instruction,
    tools=tools,
)
```


## Agent 실행 예시
Session을 생성하고 Runner를 사용해서 agent를 실행함

### Session, Runner 선언
```
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

APP_NAME="WeatherApp"

# Initialize Agent Runner
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# Create Session
session = await session_service.create_session(
    app_name="WeatherApp",
    user_id=user_id,
)
session_id = session.id
```

### run
```
from google.genai import types as genai_types

user_content = genai_types.Content(
    role="user",
    parts=[genai_types.Part(text=query)],
)

# Runner 실행 — 이벤트 스트림에서 최종 모델 응답 추출
final_response = ""
async for event in runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=user_content,
):
    # is_final_response()가 True인 이벤트의 텍스트가 최종 답변
    if event.is_final_response() and event.content:
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                final_response += part.text
```

### stream
run_async 메서드로 다음 event를 스트리밍 받음

```
async for event in runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=user_content,
):
    print(f"[{event.author}]")
    # ── 1. 유저 이벤트 스킵 ──────────────────────────────
    if event.author == "user":
        continue

    # ── 2. Tool Call (LLM → Tool 요청) ──────────────────
    function_calls = event.get_function_calls()
    if function_calls:
        # 스트리밍 텍스트가 진행 중이었으면 줄바꿈
        if current_text:
            print()
            current_text = ""
        for call in function_calls:
            print(f"\n[tool_call] tool={call.name} args={call.args}")
        continue

    # ── 3. Tool Result (Tool → LLM 응답) ────────────────
    function_responses = event.get_function_responses()
    if function_responses:
        for resp in function_responses:
            print(f"[tool_result] tool={resp.name} result={resp.response}")
        continue

    # ── 4. 텍스트 스트리밍 / 완성 ──────────────────────
    if event.content and event.content.parts:
        for part in event.content.parts:
            if not hasattr(part, "text") or not part.text:
                continue

            if event.partial:
                # 스트리밍 청크: 이전 누적 텍스트 이후 델타만 출력
                delta = part.text[len(current_text):]
                current_text = part.text
                print(delta, end="", flush=True)
            else:
                # 완성된 텍스트
                if event.is_final_response():
                    # 스트리밍 중 final이 오면 남은 델타 처리
                    delta = part.text[len(current_text):]
                    if delta:
                        print(delta, end="", flush=True)
                    print()  # 줄바꿈
                    print(f"\n[final] {part.text}")
                    current_text = ""
                else:
                    # 중간 완성 텍스트 (non-partial, non-final)
                    print(part.text)
```