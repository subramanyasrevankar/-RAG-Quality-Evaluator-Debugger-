import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_answer(question: str, chunks: list) -> str:
    """
    Generates an answer using Gemini based only on retrieved chunks.

    Key design decision — the prompt explicitly tells Gemini:
    1. Only use the provided context
    2. Don't use outside knowledge
    3. Say 'I don't know' if answer isn't in context

    This is what separates RAG from a regular chatbot.
    """

    # Format chunks into numbered context
    context = "\n\n".join([
        f"Chunk {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the context provided below.
Do not use any outside knowledge.
If the answer is not found in the context, say exactly: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def check_faithfulness_with_llm(
    question: str,
    answer: str,
    chunks: list
) -> dict:
    """
    Uses Gemini to check if the answer is faithful to the chunks.

    This is a secondary LLM call — we ask Gemini to act as a judge.
    It scores how much of the answer is supported by the context.

    Interview talking point:
    "I use LLM-as-a-judge pattern — a second Gemini call evaluates
    the first one's output. This is how production RAG systems like
    RAGAS work."
    """

    context = "\n\n".join([
        f"Chunk {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""You are a faithfulness evaluator for RAG systems.

Given the CONTEXT, QUESTION, and ANSWER below, evaluate how faithful the answer is to the context.

Faithfulness means: every claim in the answer is supported by the context.
An unfaithful answer makes claims not found in the context (hallucination).

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Respond in this exact format:
SCORE: [a number between 0.0 and 1.0]
REASON: [one sentence explaining the score]
HALLUCINATED: [yes or no]

Nothing else."""

    try:
        response = model.generate_content(prompt)
        return parse_faithfulness_response(response.text.strip())
    except Exception as e:
        return {
            "faithfulness_score": 0.0,
            "reason": f"Error: {str(e)}",
            "hallucinated": False
        }


def parse_faithfulness_response(response: str) -> dict:
    """
    Parses the structured response from the faithfulness check.
    Handles cases where Gemini doesn't follow the format exactly.
    """
    lines = response.strip().split("\n")
    result = {
        "faithfulness_score": 0.5,
        "reason": "Could not parse response",
        "hallucinated": False
    }

    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
                result["faithfulness_score"] = round(max(0.0, min(1.0, score)), 3)
            except:
                pass
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("HALLUCINATED:"):
            value = line.replace("HALLUCINATED:", "").strip().lower()
            result["hallucinated"] = value == "yes"

    return result