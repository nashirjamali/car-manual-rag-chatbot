import os

from dotenv import load_dotenv

from car_manual_chat.chain import build_chain


def main():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in a .env file before starting the chat.")

    print("Preparing the MG ZS manual...")
    chain = build_chain()
    print("Ask about a dashboard warning. Type exit to quit.")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        response = chain.invoke(question)
        print(f"Assistant: {response.content}")
        print()
