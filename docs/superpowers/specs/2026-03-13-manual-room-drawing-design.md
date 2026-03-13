# Manual Room Drawing & Polygon Editing

## Overview

Add two new canvas interaction modes: "Draw Room" (click-to-place polygon vertices) and "Edit Polygon" (drag vertices of existing rooms). Manually created or edited rooms get `source: "user"`.

## Canvas Modes

Three modes, toggled via a toolbar above the canvas:

| Mode | Behavior |
|------|----------|
| **View** (default) | Current behavior — click to select rooms, alt+drag to pan, scroll to zoom |
| **Draw** | Click to place vertices, double-click/Enter to close polygon. Esc to cancel. Live preview line from last vertex to cursor. |
| **Edit** | Select a room, then drag its vertex handles. Small circles at each polygon point, draggable. |

## Draw Flow

1. User clicks "Draw" tool in toolbar
2. Clicks points on the canvas — each click adds a vertex, live line follows cursor
3. Double-click or Enter closes the polygon (minimum 3 vertices)
4. Inline prompt asks for room name and type
5. Room saved via `POST /api/rooms` with `source: "user"`
6. Canvas returns to View mode

## Edit Flow

1. User clicks "Edit" tool in toolbar
2. Clicks an existing room to select it
3. Vertex handles appear (draggable circles at each polygon point)
4. User drags vertices to adjust boundary
5. On mouse-up, geometry recalculated and saved via `PUT /api/rooms/{id}` with `source: "user"`

## Toolbar

Small horizontal bar above the canvas with 3 icon buttons:
- Pointer (View mode)
- Pen (Draw mode)
- Edit/move (Edit mode)

Active mode highlighted. Pan/zoom remains available in all modes via alt+drag and scroll wheel.

## Backend Changes

### New endpoint: `POST /api/rooms`

Accepts: `project_id`, `name`, `room_type`, `boundary_polygon`

Computes server-side:
- `area_px` (Shapely Polygon.area)
- `perimeter_px` (Shapely Polygon.length)
- `boundary_lengths_px` (vertex-to-vertex distances)
- `centroid` (Shapely Polygon.centroid)
- Real measurements (`area_sqm`, `perimeter_m`, `boundary_lengths_m`) if project has scale

Sets `source: "user"`, `confidence: 1.0`.

### Existing endpoint: `PUT /api/rooms/{id}`

Already recalculates all geometry when polygon is updated. Needs to also set `source: "user"` when polygon changes.

## Frontend Changes

- New `CanvasToolbar` component (3 mode buttons)
- `useFloorplanCanvas` hook gains:
  - Draw mode: vertex placement, preview line, polygon closing
  - Edit mode: vertex handles rendering, drag logic
- New `createRoom()` API function
- `RoomDetail` shows source "User" with distinct color
- New room dialog (name + type input after drawing completes)
