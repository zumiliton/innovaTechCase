import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.assistant import ArduinoAssistant


def main():
    assistant = ArduinoAssistant()

    result = assistant.ask(
        board="UNO",
        question="How many digital input/output pins does this board have?",
    )

    print("\n" + "=" * 60)
    print("BOARD:", result["board"])
    print("QUESTION:", result["question"])
    print("=" * 60)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"{source['section']} | "
            f"score={source['score']:.4f}"
        )


if __name__ == "__main__":
    main()