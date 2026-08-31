import os
import sys
from telegram import Bot
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

print(
    "[MCP] TELEGRAM_BOT_TOKEN:",
    "SET" if BOT_TOKEN else "NOT SET",
    file=sys.stderr
)

server = Server("telegram-mcp-server")



@server.list_tools()
async def list_tools():

    return [
        Tool(
            name="send_telegram_message",
            description=(
                "Send a text message to a Telegram chat. "
                "Use this tool when the user explicitly asks "
                "to send information to Telegram."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Telegram chat ID"
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to send"
                    }
                },
                "required": ["chat_id", "message"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "send_telegram_message":

        if not BOT_TOKEN:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN environment variable is not set"
            )

        chat_id = arguments["chat_id"]
        message = arguments["message"]

        bot = Bot(token=BOT_TOKEN)

        await bot.send_message(
            chat_id=chat_id,
            text=message
        )

        return [
            TextContent(
                type="text",
                text="Telegram message sent successfully."
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


async def main():

    async with stdio_server() as (read_stream, write_stream):

        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":

    import asyncio

    asyncio.run(main())