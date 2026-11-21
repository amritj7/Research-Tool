from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import os

# Reads settings from .env in the project root. The Anthropic client picks up
# ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL automatically, so pointing
# ANTHROPIC_BASE_URL at http://localhost:11434 sends requests to local Ollama.
load_dotenv()

# Model used for every request: an Ollama model (e.g. qwen3:8b) when using
# Ollama, or a Claude model (e.g. claude-sonnet-5-5) when using Anthropic.
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 2024


class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.anthropic = Anthropic()
        self.available_tools: List[dict] = []

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

                    # Tool invocation through the MCP client session
                    result = await self.session.call_tool(content.name, arguments=content.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content,
                    })

            # Stop once Claude is no longer asking for tools
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

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        # Note: path is relative, so run the client from the project root.
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "src/chatbot/research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                } for tool in response.tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
