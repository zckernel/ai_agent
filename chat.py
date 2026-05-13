import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama

from tools.weather import get_weather
from tools.calculator import calculator


llm = ChatOllama(
    #model="qwen2.5:7b",
    model="llama3.1:8b",
    base_url="http://host.docker.internal:11434",
    temperature=0,
)


tools = [
    get_weather,
    calculator,
]

template = """
You are a helpful AI assistant.

IMPORTANT:
- Respond in the SAME language as the user.
- Keep answers concise.
- Use tools when needed.

You have access to the following tools:

{tools}

Tool names:
{tool_names}

STRICT RULES:
- After receiving an Observation, you MUST write "Thought: I now know the final answer" and then "Final Answer:".
- NEVER repeat an Action after receiving an Observation.
- NEVER call a tool more than once per question.

Use the following format EXACTLY:

Question: the user question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the user

If no tool is needed:
Thought: I can answer directly.
Final Answer: the final answer to the user

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=30,
)

print("AI Agent")
print("Type /exit to quit\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "/exit":
        print("Bye!")
        break

    response = agent_executor.invoke(
        {
            "input": user_input
        }
    )

    print(f"\nAI: {response['output']}\n")