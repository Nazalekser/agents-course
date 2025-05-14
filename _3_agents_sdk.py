import asyncio
from agents import Agent, Runner, function_tool


@function_tool
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    name="Calculator",
    instructions="You are a calculator. You can add two numbers together.",
    tools=[add],
)


async def main():
    response = await Runner.run(agent, "What is 2 + 2?")
    print(response.final_output)


if __name__ == "__main__":
    asyncio.run(main())
