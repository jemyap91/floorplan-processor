# Color-Aware Room Segmentation Design

## Problem

The current pipeline detects only 17 rooms from a floorplan with 150+. The wall-based morphological detector misses most rooms because:
- Grayscale conversion destroys color zone information
- Thin/broken walls fail morphological detection
- Rooms merge when wall gaps exist

The floorplan is **color-coded**: ~10% of pixels are pastel fills (pink, yellow, lavender, green) that directly correspond to room zones. This signal is completely ignored.

## Architecture

Two parallel detection paths, merged and deduplicated:

```
Image ──┬── Color Segmenter ──── colored rooms (majority)
        │
        └── Wall Segmenter ───── unfilled rooms (corridors, shafts)
                                  (runs on areas NOT covered by color zones)
        │
        └── Merge & Deduplicate ── final room list
```

## Color Segmenter (`backend/pipeline/color_segmenter.py`)

### Step 1: Extract colored pixels

Work in RGB. Identify pixels that are colored (not white, not black, not gray):
- `max(|R-G|, |G-B|, |R-B|) > 15` → pixel has color
- Exclude near-white (`R>245 & G>245 & B>245`) and near-black (`R<25 & G<25 & B<25`)

### Step 2: Quantize and label connected components

- Quantize RGB to reduce unique colors (divide by 24, round)
- For each unique quantized color, create a binary mask
- Run `cv2.connectedComponentsWithStats` on each color mask
- Each connected component = one candidate room zone

### Step 3: Split same-color rooms using wall lines

Adjacent rooms sometimes share the same pastel color. Detect dark wall lines between them:
- Extract wall lines: pixels where all channels < 80 (dark lines on the original image)
- Dilate wall lines slightly (3px kernel) to ensure they form barriers
- Before running connected components in Step 2, subtract the wall line mask from each color mask
- This splits same-color zones that are separated by wall lines

### Step 4: Filter and extract polygons

- Minimum area: 0.05% of image area (filters furniture/text colored pixels)
- Maximum area: 40% of image (filters background)
- Find contour of each component, simplify with Douglas-Peucker (tolerance=3.0)
- Build Shapely polygon, compute area/perimeter/centroid/edge lengths
- Return same dict format as existing `segment_rooms()`

## Wall Segmenter for Unfilled Rooms

Run the existing wall-based pipeline but with a twist:
- Create a "filled mask" from all color-detected room polygons
- Subtract this mask from the binary image before wall detection
- The remaining white gaps (corridors, lift shafts, lobbies) become clearly enclosed
- Run existing `segment_rooms()` on this modified wall mask
- Filter out any rooms that overlap >50% with color-detected rooms

## Merge & Deduplicate

- Start with all color-segmented rooms
- Add wall-segmented rooms that don't overlap >50% with any color room
- Sort by area descending for consistent ordering

## Integration

### New function signature

```python
def segment_rooms_by_color(
    image: np.ndarray,           # Original RGB image
    min_color_diff: int = 15,    # Channel difference threshold
    min_area_ratio: float = 0.0005,
    wall_darkness: int = 80,     # Dark pixel threshold for walls
    simplify_tolerance: float = 3.0,
    excluded_regions: list | None = None,
) -> list[dict]
```

### Pipeline changes in `main.py`

Both `_process_gemini_mode` and `_process_hybrid_mode` gain:
```python
# Color-based room detection (primary)
color_rooms = segment_rooms_by_color(image, excluded_regions=excluded_bboxes)

# Wall-based room detection for unfilled areas (secondary)
filled_mask = build_filled_mask(color_rooms, image.shape[:2])
# ... existing wall detection with filled_mask subtracted ...
wall_rooms = segment_rooms(modified_wall_mask)

# Merge
raw_rooms = merge_room_lists(color_rooms, wall_rooms)
```

## Test Plan

- Unit tests with synthetic colored rectangles on white background
- Unit tests with same-color adjacent rooms separated by dark lines
- Unit tests verifying unfilled room detection after color masking
- Integration test comparing room count before/after on sample PDF

## Expected Impact

- Color segmenter should detect 80-120 rooms from the colored zones
- Wall segmenter on remaining gaps should add 20-40 more (corridors, lifts, lobbies)
- Total: 100-160 rooms vs current 17
