# Student AI Agent

An AI agent built with LangChain that combines a fine-tuned machine translation model, Retrieval-Augmented Generation (RAG), and Model Context Protocol (MCP).

## Features

- English → Ukrainian machine translation using a fine-tuned mBART model
- RAG-based question answering about the student
- MCP integration
- Telegram bot interface
- Docker containerization
- OpenAI-powered agent orchestration

## Architecture

                    Telegram
                       |
                       v
                  Telegram Bot
                       |
                       v
                LangChain Agent
                 /      |      \
                /       |       \
               v        v        v
        Fine-tuned     RAG       MCP
          mBART       FAISS    Servers
               \        |        /
                \       |       /
                 v      v      v
                    Response
                       |
                       v
                    Telegram


## Requirements

Python 3.12
Docker
Docker Compose
OpenAI API key
Telegram Bot token

## Configuration

Creating .env file is necessary, .env.example:
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

## DOCKER run

docker compose up --build

## Local run

Install dependencies in project\agent:
pip install -r requirements.txt

Then run:
python telegram_bot.py