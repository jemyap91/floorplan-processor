"""Pure Gemini 2.5 Pro floorplan room extraction — no OpenCV pipeline."""
import json
import os
import sys

import fitz  # PyMuPDF
from PIL import Image
import io

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Try loading from .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY=") and not line.startswith("#"):
                    GOOGLE_API_KEY = line.split("=", 1)[1]

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not set")
    sys.exit(1)


PROMPT = """You are an expert architectural floor plan analyst. Analyze this floor plan image and extract every distinct room/space.

For EACH room or enclosed space you can identify, provide:
1. **name**: The room label as written on the plan (e.g. "LIFT 1", "PUBLIC LIFT LOBBY", "STAIR CORE 01"). If no label is visible, describe it (e.g. "Unlabeled corridor", "Unlabeled room near LIFT 3").
2. **type**: One of: office, bathroom, corridor, meeting_room, kitchen, storage, lobby, elevator, stairwell, utility, mechanical, parking, terrace, other
3. **boundary**: An array of [x, y] vertex points defining the room polygon, in normalized coordinates (0.0 to 1.0 where 0,0 is top-left of the FLOORPLAN area, 1,1 is bottom-right). Trace the room boundary as accurately as possible with straight-line segments.
4. **confidence**: Your confidence in the identification (0.0 to 1.0)

IMPORTANT:
- ONLY include rooms that are part of the actual building floor plan
- EXCLUDE title blocks, legends, notes, revision tables, key plans, logos, and border annotations
- Include ALL rooms you can see — corridors, lift shafts, lobbies, stairwells, plant rooms, etc.
- For each room, trace the boundary polygon carefully following the wall lines
- Use enough vertices to capture the room shape accurately (rectangles need 4, L-shapes need 6, etc.)

Return ONLY valid JSON in this exact format:
```json
{
  "rooms": [
    {
      "name": "PUBLIC LIFT LOBBY",
      "type": "lobby",
      "boundary": [[0.3, 0.4], [0.5, 0.4], [0.5, 0.6], [0.3, 0.6]],
      "confidence": 0.9
    }
  ],
  "scale_text": "1:200 or whatever scale text you can read from the drawing, or null if not found",
  "notes": "any relevant observations about the floorplan"
}
```"""


def extract_image_from_pdf(pdf_path: str, page_num: int = 0, dpi: int = 200) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def call_gemini(image: Image.Image, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

    models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
    last_err = None
    for model_name in models:
        try:
            print(f"  Trying {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, image],
                generation_config={"temperature": 0.1, "max_output_tokens": 65536},
            )
            print(f"  Success with {model_name}")
            return response.text
        except Exception as e:
            last_err = e
            print(f"  {model_name} failed: {type(e).__name__}")
            continue
    raise last_err


def parse_response(text: str) -> dict | None:
    import re
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Handle truncated response — find the JSON start and repair
    json_start = text.find('{"rooms"')
    if json_start == -1:
        json_start = text.find("```json")
        if json_start != -1:
            json_start = text.index("\n", json_start) + 1
    if json_start != -1:
        fragment = text[json_start:].rstrip("`\n ")
        # Close any open arrays/objects to make it parseable
        # Find the last complete room entry
        last_complete = fragment.rfind("},")
        if last_complete != -1:
            repaired = fragment[: last_complete + 1] + "]}"
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "input sample.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    print(f"Extracting image from {pdf_path}...")
    image = extract_image_from_pdf(pdf_path)
    print(f"Image size: {image.width}x{image.height}")

    print("Sending to Gemini 2.5 Pro...")
    raw_response = call_gemini(image, PROMPT)

    # Save raw response for debugging
    with open("gemini_raw_response.txt", "w") as f:
        f.write(raw_response)
    print("Raw response saved to gemini_raw_response.txt")

    result = parse_response(raw_response)
    if not result:
        print("Failed to parse response as JSON")
        print("First 500 chars:", raw_response[:500])
        sys.exit(1)

    rooms = result.get("rooms", [])
    print(f"\n{'='*60}")
    print(f"DETECTED {len(rooms)} ROOMS")
    print(f"{'='*60}")

    for i, room in enumerate(rooms):
        name = room.get("name", "?")
        rtype = room.get("type", "?")
        confidence = room.get("confidence", 0)
        n_vertices = len(room.get("boundary", []))
        print(f"  {i+1:3d}. {name:<40s} type={rtype:<15s} conf={confidence:.1f}  vertices={n_vertices}")

    if result.get("scale_text"):
        print(f"\nScale: {result['scale_text']}")
    if result.get("notes"):
        print(f"Notes: {result['notes']}")

    # Save structured result
    output_path = "gemini_rooms.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {output_path}")


if __name__ == "__main__":
    main()
