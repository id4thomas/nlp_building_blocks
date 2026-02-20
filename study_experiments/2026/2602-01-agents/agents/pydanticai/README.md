# PydanticAI Agent
[PydanticAI](https://ai.pydantic.dev)

## 프레임워크 파악
함수형으로 Agent를 정
- pydantic 모델을 통한 structured output 지원

핵심 기능
- Structured Output: result_type으로 pydantic 모델 지정하면 llm 응답을 자동으로 파싱, 검증
- Dependency Injection: deps_type으로 Agent 실행시 외부 의존성 (ex. DB 커넥션) 주입

### Agent Tool
PydanticAI에서는 [Toolset](https://ai.pydantic.dev/toolsets/)이라는 개념을 사용
- Toolset: collection of tools that can be registered with an agent
- Agent 선언 시점 / 실행 시점 (runtime)에 제공 가능

크게 2가지 방식의 Tool 존재
- FunctionToolset: Function 기반 (Python 함수를 tool로)
    - `@agent.tool` 데코레이터
    - Agent의 tools=[] 파라미터
    - FunctionToolSet
- MCP 기반 (외부)

Function 기반:
| 방식 | 설명 | 특성 |
| --- | --- | --- |
| `@agent.tool` 데코레이터 | 단일 Agent에 붙이는 툴 | 재사용성 낮음 |
| Agent의 tools 파라미터에 List[callable] | 함수를 직접 전달함 | |
| FunctionToolset | 도구를 모듈처럼 분리해서 관리할 경우 | tool 그룹핑 가능 |


## 구현 기초
### Client (Model) 선언
사용할 LLM Provider에 해당되는 Model 객체를 선언

OpenAI Compatible의 경우
- OpenAIChatModel / OpenAIResponsesModel 사용 가능
```
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

llm_client = AsyncOpenAI(
    base_url="http://.../v1",
    api_key="sk-...",
)
provider = OpenAIProvider(openai_client=llm_client)
model = OpenAIChatModel("Qwen...", provider=provider)
```

### Tool 연결
[FunctionToolset]

파이썬 함수를 tool로 사용할 경우
- 함수는 [RunContext](https://ai.pydantic.dev/api/tools/)라는 실행 컨텍스트 정보 접근이 가능

tool 선언 방식 3가지
- `@agent.tool` 데코레이터: 특정 에이전트에 등록할 tool
- tools 파라미터로 함수 넘기기
- FunctionToolset: 재사용을 위해 함수들을 한 셋으로 묶는 것

agent.tool 예시:
- RunContext가 필요 없을 경우 tool_plain 사용
```
agent = Agent(...)

@agent.tool
async def get_current_time(ctx: RunContext) -> str:
    ...

@agent.tool_plain
async def get_current_time() -> str:
    ...
```

tools 파라미터 예시:
```
def get_weather(city: str) -> str:
    ...
agent = Agent(... tools=[get_weather],)
```

FunctionToolset 예시:
```
from pydantic_ai.toolsets import FunctionToolset

my_toolset = FunctionToolset()

@my_toolset.tool
async def search_web(query: str) -> str:
    ...
agent = Agent(..., toolsets=[my_toolset, ..])
```

[MCP]

pydantic_ai.mcp의 `MCPServerStreamableHTTP`를 사용해 MCP 서버를 `toolset`으로 정의
- tool_prefix로 도구명 충돌 방지 가능
- 커스텀 httpx.AsyncClient 주입 가능


예시:
```
from pydantic_ai.mcp import MCPServerStreamableHTTP

location_toolset = MCPServerStreamableHTTP(
    url="http://.../mcp",
    http_client=httpx.AsyncClient(
        headers={"x-user-id": user_id},
        timeout=60,
        follow_redirects=True,
    ),
    tool_prefix="location",
)

toolsets = [location_toolset, ...]
```

### Agent 선언

```
from pydantic_ai import Agent

agent = Agent(
    model=model,
    model_settings = {"temperature": 0.7, ...},
    name="WeatherAgent",
    instructions=AGENT_INSTRUCTION,
    toolsets=toolsets,
    end_strategy="exhaustive"
)
```

model_settings로 생성 파라미터 전달 가능

end_strategy: tool call 완료 전략
- "early": 첫 output을 만나면 종료 (tool call 남아있어도)
- "exhaustive": tool call이 남아있으면 끝까지 실행 후 종료


최종 결과가 구조화 되어야할 경우 output_type 지정 가능
```
class WeatherReport(BaseModel):
    summary: str
    cities: list[str]
    temperature_range: str

    @field_validator('cities')
    def cities_not_empty(cls, v):
        if not v:
            raise ValueError("cities는 비어있을 수 없습니다")
        return v

agent = Agent(
    ...
    output_type=WeatherReport,
    output_retries=3
)

# Output Validator
@agent.output_validator
async def validate_weather_report(ctx: RunContext, output: WeatherReport) -> WeatherReport:
    # 조건 미충족 시 ModelRetry로 LLM에게 재생성 요청
    if len(output.cities) < 2:
        raise ModelRetry("최소 2개 이상의 도시 정보가 포함되어야 합니다. 다시 작성해주세요.")
    
    if not output.temperature_range:
        raise ModelRetry("temperature_range가 비어있습니다. 형식: '1°C ~ 6°C'")
    
    return output

```
- output_retries: output validation 실패시 재시도
- output_validator: 추가 검증 레이어
    - ModelRetry exception을 올려서 재생성 유도 가능


### Agent Dependency 주입
agent가 필요한 dependency를 주입 가능하다
- Agent 선언시 deps_type으로 타입 선언
- run 시점에 실제 dependency 데이터 주입
- prompt 구성 / tool 사용/ output validation 과정에 활용 가능


dependency 데이터 명세 등록
```
@dataclass
class MyDeps:
    user_id: str
    http_client: httpx.AsyncClient

agent = Agent(
    model=model,
    deps_type=MyDeps,  # 타입만 전달 (인스턴스 X)
)
```

agent 실행 시점에 주입
```
async with httpx.AsyncClient() as client:
    deps = MyDeps(user_id="1234", http_client=client)
    result = await agent.run(query, deps=deps)
```

prompt (system/user)에 사용
```
@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    return f"사용자 ID: {ctx.deps.user_id}"
```

tool
```
@agent.tool
async def fetch_data(ctx: RunContext[MyDeps], query: str) -> str:
    response = await ctx.deps.http_client.get(
        "https://api.example.com",
        headers={"x-user-id": ctx.deps.user_id}
    )
    return response.text
```

output validation
```
@agent.output_validator
async def validate(ctx: RunContext[MyDeps], output: str) -> str:
    # output 검증 로직에 deps 활용 가능
    ...
```

## Agent 실행 예시
### run
요청 후 최종 결과만 받을 경우
```
result: AgentRunResult = await agent.run(query)

# 생성 결과
print(result.output)

# 사용 토큰
print(result.usage())
# -> RunUsage(input_tokens=3809, output_tokens=381, requests=4, tool_calls=6)
```

### stream
iter 메서드를 통해 node 단위로 순회가 가능 (Agent 내부 그래프 대로)
- is_end_node를 통해 종료 판단 (내부에서 isinstance 체크)

```
UserPromptNode        # 사용자 입력 처리
  ↓
ModelRequestNode      # LLM 호출 (is_model_request_node)
  ↓
CallToolsNode         # tool 실행 (is_call_tools_node)
  ↓
ModelRequestNode      # tool 결과 반영 후 LLM 재호출
  ↓
CallToolsNode         # 추가 tool 실행 (있을 경우)
  ↓
  ...
  ↓
EndNode               # 종료 (is_end_node)
```


예시:
```
from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

current_text = ""

async with agent.iter(query) as run:
    async for node in run:
        # print(type(node))
        if Agent.is_model_request_node(node):
            if Agent.is_model_request_node(node):
                print("[MODEL REQUEST NODE]")
            elif Agent.is_user_prompt_node(node):
                print("[USER REQUEST NODE]")
                
            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartStartEvent):
                        # 첫 글자 여기 있음
                        if hasattr(event.part, 'content') and event.part.content:
                            current_text += event.part.content
                            print(event.part.content, end="", flush=True)

                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        delta = event.delta.content_delta
                        current_text += delta
                        print(delta, end="", flush=True)

        elif Agent.is_call_tools_node(node):
            print("[TOOL CALL NODE]")
            if current_text:
                print()
                current_text = ""

            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, FunctionToolCallEvent):
                        print(f"\n[tool_call] tool={event.part.tool_name} args={event.part.args}")
                    elif isinstance(event, FunctionToolResultEvent):
                        print(f"[tool_result] {event.result.content}")

        elif Agent.is_end_node(node):
            print("[END NODE]")
            if current_text:
                print()
                current_text = ""
            print(f"\n[final] {run.result.output}")
```

결과 예시:
```
[MODEL REQUEST NODE]
네, 현재 위치와 주변 도시 정보를 확인하고, 그 도시들의 날씨도 함께 알려드리겠습니다.
...

[TOOL CALL NODE]
## tool 요청
[tool_call] tool=weather_get_current_weather args={"city_name": "Busan"}
[tool_call] tool=weather_get_current_weather args={"city_name": "Gimhae"}
[tool_call] tool=weather_get_current_weather args={"city_name": "Changwon"}
[tool_call] tool=weather_get_current_weather args={"city_name": "Ulsan"}

## tool 결과
[tool_result] {'time': '20260219-19:00', 'status': {'temp_cel': 7, 'condition': '구름 조금', 'humidity_pct': 60, 'wind_kph': 20}}
[tool_result] {'time': '20260219-19:00', 'status': {'temp_cel': 6, 'condition': '맑음', 'humidity_pct': 59, 'wind_kph': 13}}
[tool_result] {'time': '20260219-19:00', 'status': {'temp_cel': 5, 'condition': '구름 조금', 'humidity_pct': 54, 'wind_kph': 13}}
[tool_result] {'time': '20260219-19:00', 'status': {'temp_cel': 7, 'condition': '흐림', 'humidity_pct': 60, 'wind_kph': 17}}

[END NODE]
[final] 다음은 현재 위치와 주변 도시들의 날씨 정보입니다:
...
```