import os
from dotenv import load_dotenv
load_dotenv()
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    print(f"Chat ID: {update.effective_chat.id}")

    await update.message.reply_text(
        f"Your chat ID: {update.effective_chat.id}"
    )


def main():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.run_polling()


if __name__ == "__main__":
    main()