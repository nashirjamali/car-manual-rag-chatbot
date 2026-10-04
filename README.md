# Car Manual RAG Chatbot

A terminal chatbot that answers questions about MG ZS dashboard warnings. Answers come from the owner's manual, not from the model's general knowledge.

## Input and output

**Input:** a question typed in the terminal, for example "The Gasoline Particular Filter Full warning has appeared. What should I do?"

**Output:** a short answer based on the matching section of the manual. For that warning, the manual says the gasoline particulate filter is full and the driver should consult an MG Authorised Repairer.

## How it works

1. `data/mg-zs-warning-messages.html` is loaded and split into chunks of about 1000 characters, with 200 characters of overlap.
2. Each chunk is embedded with `openai/text-embedding-3-small` and stored in a local Chroma index at `.chroma/`. The first run creates that index. Later runs reuse it.
3. A question is embedded with the same model. Chroma returns the 3 closest chunks.
4. Those chunks and the question are sent to `openai/gpt-4o-mini` through OpenRouter.

Both API calls use the key in `OPENAI_API_KEY` and the OpenRouter base URL `https://openrouter.ai/api/v1`.

## Project layout


| Path                               | Role                                                                |
| ---------------------------------- | ------------------------------------------------------------------- |
| `car_manual_chat/chain.py`         | Load the manual, build the Chroma index, and assemble the RAG chain |
| `car_manual_chat/cli.py`           | Read questions from the terminal and print answers                  |
| `main.py`                          | Starts the same chat loop                                           |
| `data/mg-zs-warning-messages.html` | MG ZS warning messages and the recommended action for each one      |




## Setup

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```bash
uv sync
cp .env.example .env
```

Put an OpenRouter API key in `.env`:

```bash
OPENAI_API_KEY=your-openrouter-key
```



## Run

```bash
uv run car-manual-chat
```

`uv run python main.py` starts the same chat. Type `exit` or `quit` to leave.