import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.session import ArduinoSession


def main():
    print("=" * 60)
    print("ARDUINO CONVERSATION TEST")
    print("=" * 60)

    session = ArduinoSession()

    # --------------------------------------------------
    # Simulate board identification
    # --------------------------------------------------

    session.board = "UNO"
    session.confidence = 0.95

    print(f"\nBoard: {session.board}")
    print(f"Confidence: {session.confidence}")

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    questions = [
        "How many digital input/output pins does this board have?",
        "Which ones support PWM?",
        "What are those pins used for?",
        "Can I use one of them to control the brightness of an LED?",
    ]

    for i, question in enumerate(questions, start=1):

        print("\n" + "-" * 60)
        print(f"QUESTION {i}")
        print("-" * 60)

        print(f"User: {question}")

        result = session.ask(question)

        print(f"\nAssistant: {result['answer']}")

        print("\nSources:")

        for source in result["sources"]:
            print(
                f"  - {source['source']} "
                f"| {source['section']} "
                f"| score={source['score']:.4f}"
            )

        print("\nConversation memory:")

        for message in session.messages:
            print(
                f"  {message['role']}: "
                f"{message['content']}"
            )


if __name__ == "__main__":
    main()