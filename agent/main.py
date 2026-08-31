import asyncio

from agent import create_student_agent


async def main():

    agent = await create_student_agent()

    print()
    print("Student AI Agent")
    print("Type 'exit' to quit")

    while True:

        message = input("\n>: ")

        if message.lower() == "exit":
            break

        print(repr(message))
        
        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        })

        print("\nAgent:")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())