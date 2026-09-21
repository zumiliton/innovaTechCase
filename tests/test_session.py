import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from PIL import Image

from app.pipeline.session import ArduinoSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGES = {
    "MEGA": PROJECT_ROOT / "data" / "test_images" / "mega.jpg",
    "UNO": PROJECT_ROOT / "data" / "test_images" / "uno.jpg",
    "NANO": PROJECT_ROOT / "data" / "test_images" / "nano.jpg",
}

QUESTIONS = {
    "MEGA": "How many digital input/output pins does this board have?",
    "UNO": "How many digital input/output pins does this board have?",
    "NANO": "How many digital input/output pins does this board have?",
}


def main():
    session = ArduinoSession()

    for expected_board, image_path in TEST_IMAGES.items():

        print("\n" + "=" * 60)
        print(f"EXPECTED BOARD: {expected_board}")
        print(f"IMAGE: {image_path.name}")
        print("=" * 60)

        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            continue

        image = Image.open(image_path)

        prediction = session.classify_image(image)

        print(f"Predicted board: {prediction['board']}")
        print(f"Confidence: {prediction['confidence']:.4f}")

        result = session.ask(
            QUESTIONS[expected_board]
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"][:2]:
            print(
                f"- {source['source']} | "
                f"{source['section']} | "
                f"score={source['score']:.4f}"
            )


if __name__ == "__main__":
    main()