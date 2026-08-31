import sys
from pathlib import Path
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.fine_tuned_tool import fine_tuned_model
from tools.rag_tool import student_rag

from config import OPENAI_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "server.py"
TELEGRAM_MCP_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "telegram_server.py"


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)


async def create_student_agent():

    mcp_client = MultiServerMCPClient(
        {
            "student_mcp": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(MCP_SERVER_PATH)
                ],
            },

            "telegram_mcp": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(TELEGRAM_MCP_SERVER_PATH)
                ],
                "env": {
                    "TELEGRAM_BOT_TOKEN": os.environ["TELEGRAM_BOT_TOKEN"]
                },
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()

    print("MCP tools:")

    for tool in mcp_tools:
        print(f" - {tool.name}")

    tools = [
        fine_tuned_model,
        student_rag,
        *mcp_tools
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are a helpful AI assistant.

You have access to three types of tools.

1. fine_tuned_model:
   Use this tool for machine translation from English
   to Ukrainian using the fine-tuned machine translation model.

2. student_rag:
   Use this tool for questions about the student
   and information stored in the student knowledge base.

3. MCP tools:
   Use MCP tools when the user's request requires
   functionality provided by an MCP server.
   
4. send_telegram_message:
   Use this tool whenever the user explicitly asks you
   to send, forward, or deliver information to Telegram.

Rules:
- Use fine_tuned_model for translation requests.
- Use student_rag for questions about the student.
- Use MCP tools when appropriate.
- Do not invent information about the student.
- If student_rag says that information is unavailable,
  report that to the user.
- When you use fine_tuned_model for translation, return the tool's output exactly as provided.
- Do not rewrite, improve, correct, summarize, or regenerate the translation.
- Do not perform the translation yourself.
"""
    )

    return agent