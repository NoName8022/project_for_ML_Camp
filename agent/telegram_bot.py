import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from agent import create_student_agent
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


agent = None

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    global agent

    if not update.message:
        return

    user_message = update.message.text

    if not user_message:
        return

    print(
        f"[Telegram] "
        f"{update.effective_user.username}: "
        f"{user_message}"
    )

    try:

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        })

        messages = result["messages"]

        response = messages[-1].content

        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                if getattr(message, "name", None) == "fine_tuned_model":
                    response = message.content
                    break

        await update.message.reply_text(response)

    except Exception as e:

        print(f"[ERROR] {e}")

        await update.message.reply_text(
            "An error occurred while processing your request."
        )


async def main():

    global agent

    print("Loading agent...")

    agent = await create_student_agent()

    print("Agent loaded.")
    print("Starting Telegram bot...")

    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    print("Telegram bot is running.")

    try:

        while True:
            await asyncio.sleep(3600)

    finally:

        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())