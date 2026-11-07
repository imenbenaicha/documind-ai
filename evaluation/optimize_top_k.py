
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (
    extract_chunks,
    precision_at_k,
    recall_at_k
)
from src.documind.semantic_search import build_index, search


def main():

    pdf_path = Path(sys.argv[1])

    # Load evaluation questions
    test_path = PROJECT_ROOT / "evaluation" / "test_cases.json"

    with open(test_path, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Extract and index PDF only once
    chunks = extract_chunks(pdf_path)

    model, embeddings = build_index(chunks)

    k_values = [1, 2, 3, 4, 5, 7, 10]

    print("\nTOP_K OPTIMIZATION")
    print("=" * 65)

    experiment_results = []

    for k in k_values:

        precision_scores = []
        recall_scores = []

        for case in test_cases:

            # Evaluate retrieval only on answerable questions
            if not case["is_answerable"]:
                continue

            results = search(
                case["question"],
                chunks,
                model,
                embeddings,
                top_k=k
            )

            relevant_pages = case["relevant_pages"]

            precision = precision_at_k(
                results,
                relevant_pages,
                k
            )

            recall = recall_at_k(
                results,
                relevant_pages,
                k
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)

        # Harmonic mean of precision and recall
        if avg_precision + avg_recall > 0:
            f1 = (
                2 * avg_precision * avg_recall
                / (avg_precision + avg_recall)
            )
        else:
            f1 = 0

        experiment_results.append({
            "k": k,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1
        })

        print(
            f"K={k:2} | "
            f"Precision={avg_precision:.3f} | "
            f"Recall={avg_recall:.3f} | "
            f"F1={f1:.3f}"
        )

    # Select K with the highest F1
    best_result = max(
        experiment_results,
        key=lambda result: result["f1"]
    )

    print("\n" + "=" * 65)
    print("BEST RESULT")
    print("=" * 65)

    print(f"Optimal K: {best_result['k']}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall: {best_result['recall']:.3f}")
    print(f"F1: {best_result['f1']:.3f}")

    output_path = (
        PROJECT_ROOT / "evaluation" / "top_k_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
