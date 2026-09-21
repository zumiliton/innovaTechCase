from app.pipeline.assistant import ArduinoAssistant
from app.vision.classifier import ArduinoClassifier


class ArduinoSession:

    def __init__(self):

        self.classifier = ArduinoClassifier()

        self.assistant = ArduinoAssistant()

        self.board = None

        self.confidence = None

        self.messages = []

    # --------------------------------------------------
    # Board identification
    # --------------------------------------------------

    def classify_image(self, image):

        prediction = self.classifier.predict(image)

        self.board = prediction["board"]

        self.confidence = prediction["confidence"]

        # New board = new conversation
        self.messages = []

        return prediction

    # --------------------------------------------------
    # Chat
    # --------------------------------------------------

    def ask(self, question: str):

        if self.board is None:

            raise ValueError(
                "No Arduino board has been identified yet."
            )

        # Generate answer using previous conversation
        result = self.assistant.ask(
            board=self.board,
            question=question,
            history=self.messages,
        )

        # Save user message
        self.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Save assistant response
        self.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )

        return {
            "board": self.board,
            "confidence": self.confidence,
            "answer": result["answer"],
            "sources": result["sources"],
        }