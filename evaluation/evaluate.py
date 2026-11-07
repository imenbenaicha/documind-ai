
import argparse
import json
import re
import sys

from pathlib import Path
from pypdf import PdfReader

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.documind.semantic_search import (
    build_index,
    search,
    adaptive_search
)
from documind.generation_old import (
    generate_answer,
    evaluate_faithfulness
)


# ============================================================
# 1. PDF EXTRACTION AND CHUNKING
# ============================================================

def extract_chunks(pdf_path, chunk_size=180, overlap=40):

    reader = PdfReader(pdf_path)
    chunks = []

    step = chunk_size - overlap

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""
        words = text.split()

        for start in range(0, len(words), step):

            passage = " ".join(
                words[start:start + chunk_size]
            )

            if passage.strip():
                chunks.append({
                    "page": page_number,
                    "text": passage
                })

            if start + chunk_size >= len(words):
                break

    return chunks


# ============================================================
# 2. EVALUATION METRICS
# ============================================================

def precision_at_k(results, relevant_pages):

    if not results:
        return 0.0

    relevant_count = sum(
        1 for result in results
        if result["page"] in relevant_pages
    )

    return relevant_count / len(results)


def recall_at_k(results, relevant_pages):

    if not relevant_pages:
        return None

    retrieved_pages = {
        result["page"] for result in results
    }

    relevant_retrieved = (
        retrieved_pages & set(relevant_pages)
    )

    return len(relevant_retrieved) / len(relevant_pages)


def normalize_answer(text):

    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()


def check_answer_correctness(answer, expected_answer):

    if expected_answer is None:
        return None

    answer_normalized = normalize_answer(answer)
    expected_normalized = normalize_answer(expected_answer)

    return expected_normalized in answer_normalized


# ============================================================
# 3. EVALUATE ONE QUESTION WITH ONE STRATEGY
# ============================================================

def evaluate_question(
    test_case,
    chunks,
    model,
    embeddings,
    strategy
):

    question = test_case["question"]
    relevant_pages = test_case["relevant_pages"]
    expected_answer = test_case["expected_answer"]
    expected_answerable = test_case["is_answerable"]

    # Select retrieval strategy
    if strategy == "fixed":

        results = search(
            question,
            chunks,
            model,
            embeddings,
            top_k=3
        )

    elif strategy == "adaptive":

        results = adaptive_search(
            question,
            chunks,
            model,
            embeddings,
            max_k=5,
            relevance_threshold=0.35
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    retrieved_pages = [
        result["page"] for result in results
    ]

    precision = precision_at_k(
        results,
        relevant_pages
    )

    recall = recall_at_k(
        results,
        relevant_pages
    )

    # Generate answer using the retrieved evidence
    answer_result = generate_answer(
        question,
        results
    )

    generated_answer = answer_result["answer"]
    predicted_answerable = answer_result["is_answerable"]

    faithfulness_result = evaluate_faithfulness(
    question,
    results,
    generated_answer
)

    faithfulness_score = faithfulness_result["faithfulness_score"]
    faithfulness_explanation = faithfulness_result["explanation"]

    answerability_correct = (
        predicted_answerable == expected_answerable
    )

    answer_correct = check_answer_correctness(
        generated_answer,
        expected_answer
    )

    return {
        "id": test_case["id"],
        "question": question,
        "strategy": strategy,
        "retrieved_chunks": len(results),
        "retrieved_pages": retrieved_pages,
        "similarity_scores": [
            round(result["score"], 3)
            for result in results
        ],
        "precision": precision,
        "recall": recall,
        "generated_answer": generated_answer,
        "expected_answer": expected_answer,
        "predicted_answerable": predicted_answerable,
        "expected_answerable": expected_answerable,
        "answerability_correct": answerability_correct,
        "answer_correct": answer_correct,
        "faithfulness_score": faithfulness_score,
        "faithfulness_explanation": faithfulness_explanation,
    }


# ============================================================
# 4. AGGREGATE METRICS
# ============================================================

def calculate_summary(results):

    # Retrieval metrics are calculated only on answerable questions.
    # The unanswerable question has no relevant pages.
    answerable_results = [
        result for result in results
        if result["expected_answerable"]
    ]

    precision_values = [
        result["precision"]
        for result in answerable_results
    ]

    recall_values = [
        result["recall"]
        for result in answerable_results
        if result["recall"] is not None
    ]

    answerability_accuracy = sum(
        result["answerability_correct"]
        for result in results
    ) / len(results)

    answer_correctness_values = [
        result["answer_correct"]
        for result in results
        if result["answer_correct"] is not None
    ]
    faithfulness_values = [
    result["faithfulness_score"]
    for result in results
]

    average_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values else None
    )

    return {
        "average_precision": (
            sum(precision_values) / len(precision_values)
            if precision_values else None
        ),
        "average_recall": (
            sum(recall_values) / len(recall_values)
            if recall_values else None
        ),
        "answerability_accuracy": answerability_accuracy,
        "answer_correctness": (
            sum(answer_correctness_values)
            / len(answer_correctness_values)
            if answer_correctness_values else None
        ),
        "average_retrieved_chunks": (
            sum(result["retrieved_chunks"] for result in results)
            / len(results)
        ),
        "average_faithfulness": average_faithfulness,
    }


# ============================================================
# 5. DISPLAY COMPARISON
# ============================================================

def display_summary(name, summary):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Average Precision: "
        f"{summary['average_precision']:.3f}"
    )

    print(
        f"Average Recall: "
        f"{summary['average_recall']:.3f}"
    )

    print(
        f"Answerability Accuracy: "
        f"{summary['answerability_accuracy']:.2%}"
    )

    if summary["answer_correctness"] is not None:
        print(
            f"Answer Correctness: "
            f"{summary['answer_correctness']:.2%}"
        )

    print(
        f"Average Retrieved Chunks: "
        f"{summary['average_retrieved_chunks']:.2f}"
    )

    if summary["average_faithfulness"] is not None:
        print(
            f"Faithfulness: "
            f"{summary['average_faithfulness']:.2%}"
        )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Compare fixed and adaptive RAG retrieval."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF document."
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    test_cases_path = (
        PROJECT_ROOT / "evaluation" / "test_cases.json"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    with open(test_cases_path, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print("\nDOCUMIND AI — RETRIEVAL STRATEGY COMPARISON")
    print("=" * 65)

    # Extract document
    print("\nExtracting PDF text...")

    chunks = extract_chunks(pdf_path)

    print(f"Extracted {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No text chunks were extracted from the PDF.")

    # Build embeddings once and reuse them for both strategies
    print("\nBuilding semantic index...")

    model, embeddings = build_index(chunks)

    print("Index created successfully.")

    all_results = {
        "fixed": [],
        "adaptive": []
    }

    # Evaluate both strategies
    for strategy in ["fixed", "adaptive"]:

        print("\n\n" + "#" * 65)
        print(f"EVALUATING: {strategy.upper()} RETRIEVAL")
        print("#" * 65)

        for test_case in test_cases:

            print(
                f"\nQuestion {test_case['id']}: "
                f"{test_case['question']}"
            )

            result = evaluate_question(
                test_case,
                chunks,
                model,
                embeddings,
                strategy
            )

            all_results[strategy].append(result)

            print("Retrieved pages:", result["retrieved_pages"])
            print("Retrieved chunks:", result["retrieved_chunks"])
            print("Similarity scores:", result["similarity_scores"])
            print("Precision:", round(result["precision"], 3))
            print("Recall:", result["recall"])
            print("Generated answer:", result["generated_answer"])
            print(
                "Answerability correct:",
                result["answerability_correct"]
            )
            print("Answer correct:", result["answer_correct"])

    # Calculate summaries
    fixed_summary = calculate_summary(all_results["fixed"])
    adaptive_summary = calculate_summary(all_results["adaptive"])

    # Display comparison
    display_summary("FIXED TOP-K = 3", fixed_summary)
    display_summary("ADAPTIVE RETRIEVAL", adaptive_summary)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Metric':<30}"
        f"{'Fixed':>15}"
        f"{'Adaptive':>15}"
    )

    print("-" * 60)

    metrics = [
        ("Precision", "average_precision"),
        ("Recall", "average_recall"),
        ("Answerability Accuracy", "answerability_accuracy"),
        ("Answer Correctness", "answer_correctness"),
        ("Average Retrieved Chunks", "average_retrieved_chunks"),
        ("Faithfulness", "average_faithfulness"),
    ]

    for label, key in metrics:

        fixed_value = fixed_summary[key]
        adaptive_value = adaptive_summary[key]

        if fixed_value is None or adaptive_value is None:
            continue

        if key in [
            "answerability_accuracy",
            "answer_correctness"
        ]:
            fixed_text = f"{fixed_value:.2%}"
            adaptive_text = f"{adaptive_value:.2%}"
        else:
            fixed_text = f"{fixed_value:.3f}"
            adaptive_text = f"{adaptive_value:.3f}"

        print(
            f"{label:<30}"
            f"{fixed_text:>15}"
            f"{adaptive_text:>15}"
        )

    # Save detailed results
    output = {
        "pdf": str(pdf_path),
        "number_of_chunks": len(chunks),
        "test_cases": len(test_cases),
        "configuration": {
            "fixed_top_k": 3,
            "adaptive_max_k": 5,
            "adaptive_threshold": 0.35
        },
        "fixed_summary": fixed_summary,
        "adaptive_summary": adaptive_summary,
        "fixed_results": all_results["fixed"],
        "adaptive_results": all_results["adaptive"]
    }

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "comparison_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nDetailed comparison saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
