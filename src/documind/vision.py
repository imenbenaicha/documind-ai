
import os
import io
import json
import base64
import hashlib

import fitz
from PIL import Image
from groq import Groq, RateLimitError
import time


VISION_MODEL = "qwen/qwen3.8-27b"
MAX_IMAGE_DIMENSION = 1800


def prepare_image(image_bytes):
    """Validate, resize, and convert an image to JPEG."""
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    image = image.convert("RGB")

    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)

    return output.getvalue()


def _make_visual_record(image_bytes, page, source_type):
    """Create a visual record with a stable content identifier."""
    prepared = prepare_image(image_bytes)
    image_id = hashlib.sha256(prepared).hexdigest()

    return {
        "image_id": image_id,
        "page": page,
        "source_type": source_type,
        "image_bytes": prepared,
        "mime_type": "image/jpeg",
    }


def extract_pdf_visuals(pdf_bytes):
    """
    Extract embedded images and render every PDF page.

    Page renders also capture charts, diagrams, and other vector graphics
    that may not exist as separate embedded image files.
    """
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    visuals = []
    seen = set()

    try:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            page_number = page_index + 1

            # 1. Render the complete page to capture all visual content.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False
            )
            page_image = pixmap.tobytes("png")

            record = _make_visual_record(
                page_image,
                page_number,
                "full_page"
            )

            key = (page_number, record["image_id"])
            if key not in seen:
                seen.add(key)
                visuals.append(record)

            # 2. Extract embedded images individually.
            for image_info in page.get_images(full=True):
                xref = image_info[0]

                try:
                    extracted = document.extract_image(xref)
                    image_bytes = extracted.get("image")

                    if not image_bytes:
                        continue

                    record = _make_visual_record(
                        image_bytes,
                        page_number,
                        "embedded_image"
                    )

                    key = (page_number, record["image_id"])
                    if key not in seen:
                        seen.add(key)
                        visuals.append(record)

                except Exception as error:
                    print(
                        f"Could not extract image {xref} "
                        f"from page {page_number}: {error}"
                    )

    finally:
        document.close()

    return visuals


def prepare_uploaded_image(image_bytes):
    """Prepare a standalone uploaded image for analysis."""
    return _make_visual_record(
        image_bytes,
        page=1,
        source_type="uploaded_image"
    )


def analyze_image(image_bytes, question=None):
    """Analyze an image using Groq's vision model."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    prepared = prepare_image(image_bytes)
    encoded_image = base64.b64encode(prepared).decode("utf-8")

    prompt = """
Analyze this image carefully for document question answering.

Identify all relevant visual information, including:
- The overall subject and purpose of the image
- Text visible in the image
- Tables, rows, columns, and their values
- Charts, axes, legends, labels, and trends
- Diagrams, components, and relationships
- Objects, labels, annotations, and important details
- Important numbers, dates, percentages, and units

Do not invent unreadable text or missing values.
If something is unclear, explicitly say so.

Return valid JSON with exactly these fields:
{
  "description": "Detailed description",
  "image_type": "photo, chart, table, diagram, screenshot, scanned page, or other",
  "detected_text": [],
  "tables": [],
  "charts": [],
  "objects": [],
  "relationships": [],
  "key_findings": [],
  "uncertainties": []
}
"""

    if question:
        prompt += f"\nPay particular attention to this question: {question}"

    client = Groq(api_key=api_key)

    response = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                # Keep the complete request configuration here
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_completion_tokens=500,
                response_format={"type": "json_object"}
            )
            break

        except RateLimitError:
            if attempt == 2:
                raise

            time.sleep(5 * (attempt + 1))

    if response is None:
        raise RuntimeError("Vision analysis did not return a response.")

    content = response.choices[0].message.content
    result = json.loads(content)

    return result


def visual_to_text(visual, analysis, document_id=None,
                   document_name="Uploaded image"):
    """Convert visual analysis into a searchable RAG chunk."""
    return {
        "document_id": document_id,
        "document_name": document_name,
        "page": visual["page"],
        "type": "image",
        "image_id": visual["image_id"],
        "source_type": visual["source_type"],
        "text": json.dumps(analysis, ensure_ascii=False, indent=2),
    }
