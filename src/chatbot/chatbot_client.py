from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import asyncio
import json
import os

# Load settings from .env in the project root:
#   ANTHROPIC_BASE_URL  - http://localhost:11434 for local Ollama; leave unset for Anthropic
#   ANTHROPIC_API_KEY   - any value (e.g. "ollama") for Ollama; your real key for Anthropic
#   MODEL               - e.g. qwen3:8b for Ollama, claude-sonnet-5-5 for Anthropic
load_dotenv()

BASE_URL = os.getenv("ANTHROPIC_BASE_URL")  # None -> Anthropic's API
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 2024

# Path is relative, so run the client from the project root.
SERVER_CONFIG = "src/chatbot/server-config.json"


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:

    def __init__(self):
        # One session per MCP server, plus a map from tool name -> owning session
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(base_url=BASE_URL, api_key=API_KEY)
        print(f"Using model '{MODEL}' at {BASE_URL or 'Anthropic API'}")
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]

        while True:
            response = self.anthropic.messages.create(
                max_tokens=MAX_TOKENS,
                model=MODEL,
                tools=self.available_tools,  # tools exposed to the LLM
                messages=messages,
            )

            # Keep the full assistant turn (text + tool_use blocks) in the history
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for content in response.content:
                if content.type == "text":
                    print(content.text)
                elif content.type == "tool_use":
                    print(f"Calling tool {content.name} with args {content.input}")

                    # Find the server session that owns this tool and call it there
                    session = self.tool_to_session.get(content.name)
                    if session is None:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": f"Unknown tool: {content.name}",
                            "is_error": True,
                        })
                        continue

                    result = await session.call_tool(content.name, arguments=content.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content,
                    })

            # Stop once the model is no longer asking for tools
            if response.stop_reason != "tool_use":
                break

            # Send back one result for every tool_use block, all in a single message
            messages.append({"role": "user", "content": tool_results})

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_servers(self):
        """Connect to all MCP servers listed in server-config.json."""
        try:
            with open(SERVER_CONFIG, "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server and register its tools."""
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def cleanup(self):
        """Close all server connections."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # Sessions aren't opened with "async with", so close them in cleanup()
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
