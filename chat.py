import asyncio
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


async def _ask_agent_async(user_input: str, chat_history: str) -> str:
    t0 = time.time()
    output = ""
    buffer = ""
    streaming = False
    printed_prefix = False

    try:
        async for event in agent_executor.astream_events(
            {"input": user_input, "chat_history": chat_history},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_start":
                buffer = ""
                streaming = False

            elif kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if not token:
                    continue
                buffer += token

                if not streaming:
                    if "Final Answer:" in buffer:
                        streaming = True
                        after = buffer.split("Final Answer:", 1)[1].lstrip()
                        if after:
                            if not printed_prefix:
                                print("\nAI: ", end="", flush=True)
                                printed_prefix = True
                            print(after, end="", flush=True)
                            output = after
                else:
                    if not printed_prefix:
                        print("\nAI: ", end="", flush=True)
                        printed_prefix = True
                    print(token, end="", flush=True)
                    output += token

            elif kind == "on_chain_end" and event["name"] == "AgentExecutor":
                chain_out = event["data"].get("output", {})
                final = chain_out.get("output", "") if isinstance(chain_out, dict) else str(chain_out)
                if not output:
                    print(f"\nAI: {final}", end="", flush=True)
                    output = final

    except Exception as e:
        print(f"\n[error] {e}")
        output = "Не вдалося отримати відповідь. Спробуйте переформулювати запит."

    print(f"\n[perf] agent total: {time.time() - t0:.2f}s")
    return output.strip()


def ask_agent(user_input: str, chat_history: str) -> str:
    return asyncio.run(_ask_agent_async(user_input, chat_history))


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
            print()

        chat_history += f"Human: {user_input}\nAI: {ai_output}\n"


if __name__ == "__main__":
    main()
