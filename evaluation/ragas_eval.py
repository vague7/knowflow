"""RAGAS evaluation script — runs retrieve + answer on test cases and scores with RAGAS metrics."""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from agents.answer import generate_answer
from agents.retriever import retrieve

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def run_evaluation():
    """Run RAGAS evaluation on test data."""
    # Load test data
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data.json")
    with open(test_data_path, "r") as f:
        test_cases = json.load(f)

    # Collect results
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Running pipeline on test cases...\n")
    for i, case in enumerate(test_cases):
        question = case["question"]
        print(f"  [{i + 1}/{len(test_cases)}] {question}")

        chunks = retrieve(question)
        result = generate_answer(question, chunks)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(case["ground_truth"])

    # Build dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Setup RAGAS with Gemini
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=GEMINI_KEY,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_KEY,
        )
    )

    # Assign LLM and embeddings to metrics
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    context_precision.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    # Run evaluation
    print("\nRunning RAGAS evaluation...\n")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # Print results table
    df = scores.to_pandas()
    print("=" * 80)
    print(f"{'Question':<45} {'Faith':>8} {'Relevancy':>10} {'Precision':>10}")
    print("-" * 80)

    for _, row in df.iterrows():
        q = row["question"][:42] + "..." if len(row["question"]) > 45 else row["question"]
        print(f"{q:<45} {row.get('faithfulness', 'N/A'):>8.3f} "
              f"{row.get('answer_relevancy', 'N/A'):>10.3f} "
              f"{row.get('context_precision', 'N/A'):>10.3f}")

    print("-" * 80)
    print(f"{'AVERAGE':<45} {df.get('faithfulness', [0]).mean():>8.3f} "
          f"{df.get('answer_relevancy', [0]).mean():>10.3f} "
          f"{df.get('context_precision', [0]).mean():>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
