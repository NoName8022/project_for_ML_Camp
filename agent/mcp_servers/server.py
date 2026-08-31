from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


server = Server("student-mcp-server")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_student_project_info",
            description="Get information about the student's current AI project.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "get_student_project_info":

        return [
            TextContent(
                type="text",
                text=(
                    "The student is working on an AI agent project "
                    "that uses a fine-tuned mBART translation model, "
                    "RAG, LangChain and MCP."
                ),
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


async def main():

    async with stdio_server() as (read_stream, write_stream):

        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())