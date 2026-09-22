import os
from pathlib import Path

from app.llm.local import LocalLLMProvider
from app.llm.api import APILLMProvider
from app.rag.retriever import Retriever


#PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "data" / "vectorstore" / "index.faiss"

METADATA_PATH = PROJECT_ROOT / "data" / "vectorstore" / "metadata.json"

EMBEDDING_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "embeddings"
    / "all-MiniLM-L6-v2"
)


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are a technical assistant specialized in Arduino boards.

VISUAL IDENTIFICATION
---------------------
A physical image of the board was provided by the user and has been
processed by a visual identification system.

The identified board is:
{board}

IMPORTANT:
The visual identification and the retrieved documentation are two
different sources of information.

- The image represents the physical Arduino board being discussed.
- The identified board comes from visual analysis of that physical image.
- The documentation below was retrieved by the RAG system to provide
  technical, inventory, and other textual information about the identified
  board.
- A document such as a YAML file, PDF, text file, or technical specification
  is NOT the physical image of the board.
- The absence of an image or photograph inside the retrieved documentation
  does NOT mean that the user did not provide an image.
- Do not claim that the user uploaded a document instead of an image merely
  because the RAG retrieved a document describing the board.

ANSWERING RULES
---------------
- The identified board is fixed for this conversation.
- Treat the visual identification as the identification of the physical
  object being discussed.
- Use only documentation belonging to the identified board.
- Do not use information from other Arduino board families.
- Use the retrieved documentation for technical specifications, inventory,
  availability, compatibility, and other factual information.
- Do not invent specifications or information that is not supported by the
  provided documentation.
- If a technical specification cannot be found in the provided documentation,
  explicitly say that the information is not available in the provided
  documentation.
- Do not infer that information is unavailable simply because it is not
  visible in the image.
- Do not infer that an image was not provided simply because the retrieved
  documentation is a text-based file.
- Use the previous conversation to understand follow-up questions and
  references such as "it", "those", "that board", "that pin", etc.
- Be concise and technically accurate.
- Answer in the same language as the user's question.

DOCUMENTATION RETRIEVED BY RAG
------------------------------
{context}

PREVIOUS CONVERSATION
---------------------
{conversation}

CURRENT QUESTION
----------------
{question}

ANSWER
------
"""


# ARDUINO ASSISTAN

class ArduinoAssistant:

    def __init__(self):

        # ----------------------------------------------------
        # Select LLM provider from environment
        # ----------------------------------------------------

        provider = os.getenv("LLM_PROVIDER", "local").lower()

        print(f"LLM provider: {provider}")

        if provider == "local":
            print("Creating LocalLLMProvider")
            self.llm = LocalLLMProvider()

        elif provider == "api":
            print("Creating APILLMProvider")
            self.llm = APILLMProvider()

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider}"
            )

        # ----------------------------------------------------
        # Initialize RAG retriever
        # ----------------------------------------------------

        self.retriever = Retriever(
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            embedding_model_path=EMBEDDING_MODEL_PATH,
        )


    # ASKING

    def ask(
        self,
        board: str,
        question: str,
        history: list | None = None,
        top_k: int = 5,
    ) -> dict:

        # ----------------------------------------------------
        # Initialize history
        # ----------------------------------------------------

        if history is None:
            history = []


        # ----------------------------------------------------
        # Retrieve documentation for the identified board
        # ----------------------------------------------------

        results = self.retriever.retrieve(
            question=question,
            board=board,
            top_k=top_k,
        )


        # ----------------------------------------------------
        # Build RAG context
        # ----------------------------------------------------

        if results:

            context = "\n\n".join(
                result["text"]
                for result in results
            )

        else:

            context = (
                "No documentation was retrieved for the identified board."
            )


        # ----------------------------------------------------
        # Build conversation history
        # ----------------------------------------------------

        conversation_parts = []

        for message in history:

            role = message.get("role", "").upper()

            content = message.get("content", "")

            if role and content:

                conversation_parts.append(
                    f"{role}: {content}"
                )


        if conversation_parts:

            conversation = "\n".join(
                conversation_parts
            )

        else:

            conversation = "No previous conversation."


        # ----------------------------------------------------
        # Build final prompt
        # ----------------------------------------------------

        prompt = SYSTEM_PROMPT.format(
            board=board,
            context=context,
            conversation=conversation,
            question=question,
        )


        # Debug information

        print("=" * 60)
        print("ARDUINO ASSISTANT")
        print(f"Board: {board}")
        print(f"Question: {question}")
        print(f"RAG results: {len(results)}")
        print("=" * 60)


        # Generate answer

        answer = self.llm.generate(prompt)


        # Return response

        return {
            "board": board,
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": result["metadata"]["source"],
                    "section": result["metadata"]["section"],
                    "score": result["score"],
                }
                for result in results
            ],
        }
