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

## Project Structure

D:\UNIVERSITY\PRACTICE\FINE TUNE\PROJECT
│   .gitignore
│   config.py
│   data.py
│   eda.py
│   evaluate_checkpoint.py
│   evaluation.py
│   filtering.py
│   lid.176.ftz
│   main.py
│   metrics.py
│   model.py
│   prepare_dataset.py
│   tokenizer.py
│   train.py
│
├───agent
│   │   .dockerignore
│   │   .env
│   │   agent.py
│   │   config.py
│   │   docker-compose.yml
│   │   Dockerfile
│   │   env.example
│   │   main.py
│   │   rag_service.py
│   │   README.md
│   │   requirements.txt
│   │   telegram_bot.py
│   │   __init__.py
│   │
│   ├───mcp_servers
│   │       server.py
│   │       telegram_id.py
│   │       telegram_server.py
│   │       __init__.py
│   │
│   ├───model
│   │   └───checkpoint-31250
│   │           adapter_config.json
│   │           adapter_model.safetensors
│   │           optimizer.pt
│   │           README.md
│   │           rng_state_0.pth
│   │           rng_state_1.pth
│   │           scaler.pt
│   │           scheduler.pt
│   │           tokenizer.json
│   │           tokenizer_config.json
│   │           trainer_state.json
│   │           training_args.bin
│   │
│   ├───rag
│   │   │   __init__.py
│   │   │
│   │   ├───documents
│   │   │       student.txt
│   │   │
│   │   └───vectorstore
│   │           index.faiss
│   │           index.pkl
│   │
│   ├───tools
│   │   │   fine_tuned_tool.py
│   │   │   rag_tool.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           fine_tuned_tool.cpython-312.pyc
│   │           rag_tool.cpython-312.pyc
│   │           __init__.cpython-312.pyc
│   │
│   └───__pycache__
│           agent.cpython-312.pyc
│           config.cpython-312.pyc
│           rag.cpython-312.pyc
│           tools.cpython-312.pyc
│
├───checkpoints
│   └───checkpoint-31250
│           adapter_config.json
│           adapter_model.safetensors
│           optimizer.pt
│           README.md
│           rng_state_0.pth
│           rng_state_1.pth
│           scaler.pt
│           scheduler.pt
│           tokenizer.json
│           tokenizer_config.json
│           trainer_state.json
│           training_args.bin
│
├───dataset
│   ├───test
│   │       data-00000-of-00001.arrow
│   │       dataset_info.json
│   │       state.json
│   │
│   ├───train
│   │       cache-0a8a73cc66ed8802.arrow
│   │       cache-5a658232d94a6d96.arrow
│   │       cache-8f7c81c9baabbb5b.arrow
│   │       cache-d82f628b21d9c349.arrow
│   │       cache-ddde76656b89e3be.arrow
│   │       data-00000-of-00001.arrow
│   │       dataset_info.json
│   │       state.json
│   │
│   └───validation
│           cache-0272d6209fb3b567.arrow
│           cache-091fbed19532a54b.arrow
│           cache-098485789856f174.arrow
│           cache-5962f77e01b802a5.arrow
│           cache-736f5775345d741c.arrow
│           cache-e873b058e2939d0d.arrow
│           data-00000-of-00001.arrow
│           dataset_info.json
│           state.json
│
├───final_model
│       adapter_config.json
│       adapter_model.safetensors
│       README.md
│       tokenizer.json
│       tokenizer_config.json
│       training_args.bin
│
└───__pycache__
        config.cpython-312.pyc
        data.cpython-312.pyc
        evaluation.cpython-312.pyc
        filtering.cpython-312.pyc
        metrics.cpython-312.pyc
        model.cpython-312.pyc
        tokenizer.cpython-312.pyc
        train.cpython-312.pyc

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