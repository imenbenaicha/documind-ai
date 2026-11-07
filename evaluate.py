
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from app import extract_chunks
from config import TOP_K
from src.documind.semantic_search import build_index, search
from src.documind.generation import generate_answer


def load_pdf(path):
    with open(path, "rb") as pdf_file:
        return extract_chunks(pdf_file)


def normalize_text(text):
    """Normalize formatting differences for basic answer checks."""
    text = text.lower()
    text = re.sub(r"\*\*", "", text)  # Remove Markdown bold
    text = re.sub(r"\s+", "", text)  # Remove whitespace
    return text


def evaluate_answer(answer, expected, answerable):
    """Perform basic automated checks on a generated answer."""
    normalized_answer = normalize_text(answer)

    if answerable:
        normalized_expected = normalize_text(expected)
        return normalized_expected in normalized_answer

    refusal_terms = [
    "notprovided",
    "doesnotprovide",
    "donotprovide",
    "donotprovideanyinformation",
    "notmentioned",
    "insufficientinformation",
    "cannotdetermine",
    "don'thave",
    "donothave",
    "noprofitinformation",
    "noinformation",
    "notavailable",
    "no net profit data",
    "donotincludeanyinformation",
    "donotcontainanyinformation"
]


    return any(term in normalized_answer for term in refusal_terms)


def main():
    pdf_path = "documind_test_document.pdf"

    chunks = load_pdf(pdf_path)
    model, embeddings = build_index(chunks)

    test_cases = [
        {
            "question": "What was the company's revenue in Q2 2026?",
            "expected": "$1,250,000",
            "answerable": True,
        },
        {
            "question": "What was the percentage increase in revenue from Q1 to Q2?",
            "expected": "13.6%",
            "answerable": True,
        },
        {
            "question": "What percentage of requests were resolved within one business day?",
            "expected": "90%",
            "answerable": True,
        },
        {
            "question": "What was the company's net profit in Q2 2026?",
            "expected": "not provided",
            "answerable": False,
        },
    ]

    passed = 0

    print(f"Indexed {len(chunks)} chunks.")

    for test in test_cases:
        results = search(
            test["question"],
            chunks,
            model,
            embeddings,
            top_k=TOP_K,
        )

        print(f"\nQuestion: {test['question']}")

        if not results:
            answer = "No relevant passages were retrieved."
        else:
            try:
                answer = generate_answer(test["question"], results)
            except Exception as error:
                print(f"LLM error: {error}")
                print("Result: ERROR")
                print("-" * 60)
                continue

        correct = evaluate_answer(
            answer,
            test["expected"],
            test["answerable"],
        )

        passed += int(correct)

        print(f"Expected: {test['expected']}")
        print(f"Generated answer: {answer}")
        print(f"Result: {'PASS' if correct else 'FAIL'}")
        print("-" * 60)

    print(f"\nAnswer evaluation: {passed}/{len(test_cases)} passed")
    print(f"Pass rate: {passed / len(test_cases):.1%}")


if __name__ == "__main__":
    main()
