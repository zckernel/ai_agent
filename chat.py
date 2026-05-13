import os
import time

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from tools.weather import get_weather
from tools.calculator import calculator
from tools.file_reader import WORK_DIR, read_file_content
from tools.web_search import search_news, search_wiki


# ── Constants ────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL    = "qwen2.5:14b"

FILE_EXTENSIONS = (".md", ".txt", ".py", ".json", ".csv", ".yaml", ".yml", ".log")

AGENT_PROMPT = """
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
- After receiving an Observation, you MUST immediately write EXACTLY:
  Thought: I now know the final answer
  Final Answer: [your answer here]
- NEVER repeat an Action after receiving an Observation.
- NEVER call a tool more than once per question.
- If the Observation contains file content, summarize it briefly in Final Answer.

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

Conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""


# ── LLM & Agent setup ────────────────────────────────────────────────────────

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

tools = [get_weather, calculator, search_news, search_wiki]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(AGENT_PROMPT),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,
    max_execution_time=15,
    return_intermediate_steps=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_file_path(text: str) -> str | None:
    words = text.split()
    return next(
        (w.strip('"\'') for w in words if "/" in w or "\\" in w or w.endswith(FILE_EXTENSIONS)),
        None,
    )


def looks_like_file_request(text: str) -> bool:
    return extract_file_path(text) is not None


def summarize_file(path: str, user_message: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(WORK_DIR, path)

    t0 = time.time()
    content = read_file_content.invoke(path)
    print(f"[perf] file read: {time.time() - t0:.2f}s")

    prompt = (
        f"Describe this file in 1-2 sentences: what type of file it is and what it contains. "
        f"Start with 'This is a file with...' or similar. "
        f"Do not summarize the project or topic in depth — just describe the file itself. "
        f"Respond in the same language as: '{user_message}'.\n\n"
        f"File: {path}\n\n{content}"
    )

    print("\nAI: ", end="", flush=True)
    t1 = time.time()
    chunks = []
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
        chunks.append(chunk.content)
    print(f"\n[perf] llm summarize: {time.time() - t1:.2f}s")

    return "".join(chunks)


def ask_agent(user_input: str, chat_history: str) -> str:
    t0 = time.time()
    try:
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
        output = response["output"]
        if "Agent stopped" in output and response.get("intermediate_steps"):
            last_obs = response["intermediate_steps"][-1][1]
            output = str(last_obs)
    except Exception as e:
        print(f"[error] {e}")
        output = "Не вдалося отримати відповідь. Спробуйте переформулювати запит."
    print(f"[perf] agent total: {time.time() - t0:.2f}s")
    return output


# ── Chat loop ─────────────────────────────────────────────────────────────────

def main():
    print("AI Agent")
    print("Type /exit to quit\n")

    chat_history = ""

    while True:
        user_input = input("You: ")

        if user_input.lower() == "/exit":
            print("Bye!")
            break

        if looks_like_file_request(user_input):
            path = extract_file_path(user_input)
            ai_output = summarize_file(path, user_message=user_input)
        else:
            ai_output = ask_agent(user_input, chat_history)
            print(f"\nAI: {ai_output}\n")

        chat_history += f"Human: {user_input}\nAI: {ai_output}\n"


if __name__ == "__main__":
    main()
