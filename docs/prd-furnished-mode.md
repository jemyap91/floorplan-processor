# PRD: Furnished Residential Floorplan Processing Mode

## Problem Statement

The floorplan processor currently offers three processing modes (hybrid, gemini, linedraw), but none of them can accurately detect individual rooms in **furnished residential floorplans**. When processing drawings like `input_sample_3.pdf` — a multi-unit luxury residential floor plan — the existing pipeline:

- Detects unit-level boundaries instead of individual rooms (bedrooms, bathrooms, kitchens, etc.)
- Confuses furniture (tables, beds, bathroom fixtures) with wall lines, producing false room boundaries
- Cannot segment open-concept living/dining/kitchen areas that have no separating walls
- Fails to associate rooms with their parent apartment unit

Users working with residential architectural drawings need room-level granularity — every bedroom, bathroom, kitchen, corridor, balcony, and lobby identified as a separate clickable polygon with its name, type, area, and parent unit.

## Solution

A new **"Furnished"** processing mode that uses a two-pass Gemini Vision AI approach to identify rooms semantically, bypassing the furniture noise problem entirely. Gemini can "see through" furniture because it understands that a bed means bedroom, a toilet means bathroom — something contour-based CV cannot do.

The pipeline:

1. **Pass 1 (Gemini Flash)**: Identify all apartment units and public spaces with bounding boxes. Read printed area values (m²) from text labels on the drawing.
2. **Pass 2 (Gemini Flash or Pro, user-selected)**: For each unit, crop the image to just that unit's area and ask Gemini for detailed room polygons, names, and types. Public spaces (lobbies, stairwells) skip this pass — their Pass 1 bounding box is sufficient.
3. **Wall snapping**: Refine Gemini's approximate polygon vertices by snapping each one to the nearest dark pixel (wall/ink line) within a search radius, producing cleaner polygons that align with the actual drawing.
4. **Area validation**: Calculate area from the polygon geometry and compare against the printed m² value Gemini read from the drawing. Flag significant divergences.

Each detected room is tagged with its parent unit name (e.g., "UNIT 20.01") or "Public Space", making it easy to group and filter.

Users can choose between Gemini Flash (free, fast) and Gemini Pro (more accurate, paid) for the detail-critical Pass 2, while Pass 1 always uses Flash.

## User Stories

1. As a building analyst, I want to process a furnished residential floorplan and get individual rooms detected (not just unit outlines), so that I can analyze space allocation at the room level.
2. As a building analyst, I want each detected room tagged with its parent apartment unit (e.g., "UNIT 20.01"), so that I can group rooms by unit and calculate per-unit totals.
3. As a building analyst, I want public spaces (lift lobbies, stairwells, fire hydrant rooms) detected and tagged as "Public Space", so that I can distinguish shared areas from private unit rooms.
4. As a building analyst, I want to click on any detected room polygon on the canvas, so that I can view its details (name, type, area, unit).
5. As a building analyst, I want rooms in open-concept areas (living/dining/kitchen with no walls) to be detected either as separate spaces or as a merged single space, depending on what the drawing shows, so that the output reflects the actual architectural intent.
6. As a building analyst, I want to see both the polygon-calculated area and the printed area from the drawing for each room, so that I can verify detection accuracy.
7. As a building analyst, I want rooms with significant area divergence (polygon vs. printed) flagged visually, so that I can prioritize which rooms need manual correction.
8. As a building analyst, I want to choose between Gemini Flash and Gemini Pro models, so that I can trade off between speed/cost and accuracy.
9. As a building analyst, I want the "Furnished" mode available as a button alongside the existing modes (Hybrid, Gemini, Linedraw), so that I can select the appropriate mode for my drawing type.
10. As a building analyst, I want room polygons to align closely with wall lines where walls exist, so that the visual overlay looks clean and professional on the canvas.
11. As a building analyst, I want room polygons to use Gemini's approximate boundary where no wall lines exist (open-concept areas), so that every room has a complete polygon even without physical wall separation.
12. As a building analyst, I want door openings treated as semantic room boundaries by the AI, so that rooms separated by doorways are correctly split without requiring explicit door arc detection.
13. As a building analyst, I want to see a model selector (Flash/Pro) in the UI when using Gemini-dependent modes, so that I can control the accuracy/cost tradeoff per processing run.
14. As a building analyst, I want the default model to be Flash, so that I don't accidentally incur costs when testing.
15. As a building analyst, I want to be able to manually edit, add, or delete room polygons after processing, so that I can correct any detection errors (this already exists in the app).
16. As a building analyst, I want the unit name displayed in the room sidebar and detail panel, so that I can see which unit a room belongs to without clicking through.
17. As a building analyst, I want to export rooms with their unit tags in CSV/JSON/XLSX format, so that downstream analysis tools can group by unit.
18. As a developer, I want debug images saved during processing (annotated with detected units and room polygons), so that I can diagnose issues when detection quality is poor.

## Implementation Decisions

### New Modules

**Furnished Analyzer Module** — Core Gemini two-pass orchestrator:
- Pass 1 function: accepts full image, calls Gemini Flash, returns list of unit/public-space info (name, bounding box, printed area values)
- Pass 2 function: accepts cropped unit image + unit info, calls Gemini (user-selected model), returns list of room polygons with names, types, and approximate m² values
- Orchestrator function: combines both passes, handles coordinate transformation from cropped-unit space back to full-image space
- All Gemini prompt engineering and JSON response parsing encapsulated here
- Uses normalized coordinates (0.0-1.0) in Gemini prompts, converts to pixel coordinates internally

**Wall Snapper Module** — Dark-pixel vertex snapping (v2 deliverable):
- Accepts a Shapely polygon and a grayscale/binary image
- For each vertex, searches a configurable radius (default 30-50px) for the nearest cluster of dark pixels
- Snaps vertex to the darkest/densest point within the search radius
- Returns refined polygon
- Pure geometry + image operation with no external dependencies
- Fallback plan: if snapping produces poor results, switch to aggressive erosion-based wall detection (erode furniture lines, detect remaining walls, snap to those)

### Modified Modules

**Room Data Model** — Three new fields:
- `unit_name` (optional string): parent unit identifier or "Public Space"
- `printed_area_sqm` (optional float): area value read from drawing text labels by Gemini
- `area_divergence_flag` (boolean): true when polygon-calculated and printed areas diverge significantly

**Database Layer** — Schema additions:
- Add three new columns to the rooms table: `unit_name TEXT`, `printed_area_sqm REAL`, `area_divergence_flag INTEGER`
- Use safe migration pattern (ALTER TABLE ADD COLUMN with existence check) to avoid breaking existing databases

**Backend API** — New mode routing:
- New `_process_furnished_mode()` function following the same pattern as existing mode functions
- Accept `gemini_model` parameter (string: "flash" or "pro") via Form data
- Route `mode == "furnished"` to the new function
- Pass model selection through to the furnished analyzer

**Vision AI Module** — Model selection support:
- Update the Gemini call function to accept an explicit model name parameter
- Support both `gemini-2.5-flash` and `gemini-2.5-pro` model IDs
- When no explicit model is provided, fall back to existing env var behavior (backward compatible)

**Frontend API Layer**:
- Extend the ProcessMode type to include `'furnished'`
- Add `geminiModel` parameter to the process function
- Pass model selection through FormData to the backend

**Frontend App Shell**:
- Add "Furnished" mode button to the mode selector (use a distinct color, e.g., emerald/green)
- Add model selector dropdown (Flash / Pro) that appears for Gemini-dependent modes (gemini, furnished)
- Display `unit_name` in the room sidebar list items
- Display `printed_area_sqm` and `area_divergence_flag` in the room detail panel
- Include unit name in export outputs

### Architecture Decisions

- **Gemini handles semantic understanding**: room identification, naming, typing, boundary estimation in cluttered/open-concept areas. CV handles geometric precision (wall snapping).
- **Two-pass design**: Pass 1 (lightweight, always Flash) identifies the coarse unit structure. Pass 2 (detail, user-selected model) processes one unit at a time for better coordinate accuracy and less truncation risk.
- **Public spaces skip Pass 2**: lobbies, stairwells, etc. are simple single-room spaces. Pass 1 bounding box + wall snapping is sufficient.
- **Iterative delivery**: v1 ships Gemini two-pass with raw polygons (no snapping) + debug images. v2 adds wall snapping. v3 tunes thresholds and edge cases.
- **No CV door detection**: door swing arcs are the same color as walls/furniture in these drawings. Gemini handles door semantics (understanding that a door separates two rooms) without explicit arc detection.
- **Open-concept handling**: if Gemini identifies distinct sub-spaces (kitchen vs. living room), it returns separate polygons. If Gemini sees one continuous space, it returns one merged polygon. No forced splitting.

### API Contract

The `/api/process` endpoint gains:
- `mode: "furnished"` — selects the new pipeline
- `gemini_model: "flash" | "pro"` — optional, defaults to "flash"

Room response objects gain:
- `unit_name: string | null`
- `printed_area_sqm: number | null`
- `area_divergence_flag: boolean`

## Testing Decisions

Good tests for this feature should test **external behavior and contracts**, not implementation details. They should mock Gemini API responses (no real API calls in tests) and use synthetic or fixture images for CV operations.

### Modules to Test

**Furnished Analyzer** (new, high priority):
- Test Pass 1 prompt construction produces valid prompt text
- Test Pass 1 response parsing with mocked JSON (valid, truncated, malformed)
- Test Pass 2 prompt construction includes correct room count and unit context
- Test Pass 2 response parsing with mocked JSON
- Test coordinate transformation from cropped-unit space to full-image space
- Test orchestrator combines Pass 1 + Pass 2 results correctly
- Test public spaces are returned with unit_name="Public Space" and skip Pass 2
- Prior art: `test_vision_ai.py` — same mock pattern for Gemini responses

**Wall Snapper** (new, high priority — v2):
- Test vertex snapping on synthetic binary image with known wall positions
- Test that vertices far from any dark pixel remain unchanged (beyond search radius)
- Test with empty/all-white image (no walls to snap to)
- Test that snapping preserves polygon validity (no self-intersections)
- Prior art: `test_room_segmenter.py` — uses synthetic numpy arrays for CV testing

**Room Model** (modified, low effort):
- Test new fields serialize and deserialize correctly via model_dump/model_validate
- Test default values (unit_name=None, area_divergence_flag=False)
- Prior art: `test_models.py`

**API Endpoint** (modified, medium priority):
- Test `/api/process` with `mode=furnished` routes correctly (mock the pipeline)
- Test `gemini_model` parameter is accepted and passed through
- Prior art: `test_api.py`

### Not Tested

- Frontend changes (no existing frontend test infrastructure)
- Minor wiring in `main.py` mode routing (covered by API endpoint test)
- Gemini prompt quality (subjective, validated by running against real PDFs)

## Out of Scope

- **Pixel-perfect polygon alignment**: polygons will be visually clean where walls exist but approximate in open-concept areas. Sub-pixel wall alignment is not a goal.
- **Automatic scale detection for this drawing type**: relies on existing scale detection or manual input.
- **Multi-page PDF support**: processes one page at a time (existing limitation).
- **Real-time processing**: the two-pass Gemini approach will be slower than CV-only modes. No streaming/incremental results.
- **CV-based door arc detection**: explicitly decided against. Gemini handles door semantics.
- **Unit-level aggregate views**: no UI for viewing all rooms grouped by unit in a tree/accordion. Rooms are flat-listed with unit tags.
- **Automatic model selection**: user must choose Flash vs Pro. No auto-fallback from Pro to Flash on failure (beyond existing retry logic).
- **Color zone detection**: this floorplan has no color fills. The color segmenter is not used in furnished mode.

## Further Notes

### Iteration Plan

- **v1 (initial delivery)**: Gemini two-pass pipeline producing raw polygons. No wall snapping. Debug images saved showing detected units and room polygons overlaid on the floorplan. Frontend mode button + model selector. This alone may produce surprisingly good results.
- **v2**: Add wall snapper module. Snap Gemini polygon vertices to nearest dark pixels. Measure improvement on `input_sample_3.pdf`.
- **v3**: Threshold tuning, edge case handling (very small rooms, rooms near margins, units with unusual layouts). If dark-pixel snapping is insufficient, implement aggressive erosion-based wall detection as fallback.

### Known Risks

- **Gemini coordinate accuracy**: Gemini's spatial understanding may produce polygons that are 20-50px off at 7016x9934 resolution. Wall snapping (v2) mitigates this.
- **Gemini response truncation**: large unit with many rooms may produce truncated JSON. Existing `_repair_truncated_json` logic helps. The two-pass design reduces this risk by processing one unit at a time.
- **Open-concept ambiguity**: different Gemini calls may inconsistently split or merge open-concept spaces. Prompt engineering and model choice (Pro tends to be more consistent) are the mitigations.
- **Cost**: Gemini Pro costs more. Flash is the default to keep it free-friendly. Users opt in to Pro explicitly.

### Test PDF

Primary test file: `input_sample_3.pdf` — multi-unit luxury residential floor plan ("Archer & Albert" by W=B), 7016x9934px, monochrome line drawing with dense furniture, open-concept units, blue service markers, red dashed site boundary.
