# Floorplan Processor

A desktop tool that extracts room boundaries, names, areas, and perimeters from architectural floorplan PDFs and images. Uses a hybrid pipeline combining OpenCV computer vision with Google Gemini AI for room detection and labelling.

## Features

- **Automatic room detection** from PDF or image floorplans (PNG, JPG, TIFF, BMP, WebP)
- **Hybrid CV + AI pipeline** — OpenCV detects walls and room boundaries, Gemini AI reads room labels
- **Color-based segmentation** — detects rooms with colored fills (pink, blue, green zones)
- **Scale detection** — reads scale notations (e.g. "1:200") to convert pixel measurements to real-world units
- **Manual room drawing** — click-to-place polygon tool for adding rooms the AI missed
- **Polygon editing** — drag vertices to correct room boundaries
- **Project persistence** — saves results to SQLite so you don't have to reprocess
- **Export** — download results as JSON, CSV, or Excel (.xlsx)

## Requirements

- Python 3.13 (Anaconda recommended)
- Node.js 22 (via nvm)
- A Google Gemini API key ([get one free](https://aistudio.google.com/apikey))

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd floorplan-processor

# Backend (Python)
/opt/anaconda3/bin/pip install -r requirements.txt

# Frontend (Node)
cd frontend
source ~/.nvm/nvm.sh && nvm use 22
npm install
cd ..
```

### 2. Set up environment

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-gemini-api-key-here
DB_PATH=./floorplan.db
```

See `.env.example` for reference.

### 3. Run

```bash
./run.sh
```

This starts both the backend (http://localhost:8000) and frontend (http://localhost:5173). Open the frontend URL in your browser.

## Usage Guide

### Uploading a Floorplan

1. Open http://localhost:5173 in your browser
2. Drag and drop a PDF or image file onto the upload area, or click **Select File**
3. Choose a processing mode:
   - **Gemini AI** (recommended) — CV boundaries + Gemini room labelling
   - **CV + AI** — CV boundaries + basic Gemini labelling
4. Wait for processing (progress bar shows each pipeline stage)
5. The three-panel workspace opens when done

### Workspace Layout

| Panel | Location | Purpose |
|-------|----------|---------|
| **Room Sidebar** | Left | Lists all detected rooms with area/perimeter. Click to select. |
| **Canvas** | Center | Displays the floorplan image with room polygons overlaid. |
| **Room Detail** | Right | Shows selected room's measurements, wall segments, and metadata. |

### Navigating the Canvas

- **Scroll** to zoom in/out
- **Alt + Drag** to pan
- **Click a room** to select it
- **Fit View** button resets the zoom to fit the full image

### Canvas Toolbar Modes

The toolbar at the top-left of the canvas has three modes:

#### View Mode (default)
Click rooms to select them and view their details in the right panel.

#### Draw Mode
Manually draw new room polygons:

1. Click the **Draw** button (pen icon) in the toolbar
2. Click on the canvas to place each vertex (corner) of the room
3. The vertex count is shown at the bottom of the canvas
4. **Double-click** or press **Enter** to close the polygon
5. Press **Esc** to cancel at any time
6. A dialog appears asking for the room name and type
7. Click **Create Room** — the room is saved with all measurements calculated automatically

#### Edit Mode
Reshape existing room boundaries:

1. Click the **Edit** button (move icon) in the toolbar
2. Click a room to select it — yellow vertex handles appear at each corner
3. Drag any handle to move that vertex
4. Release the mouse — the room's area, perimeter, and wall lengths are recalculated automatically
5. Press **Esc** or click empty space to deselect

Rooms created or edited manually are marked as **"User (manual)"** in the detail panel.

### Editing Room Details

In the right panel, you can:

- **Rename** a room by editing the name field
- **Change the type** (office, bathroom, corridor, etc.) via the dropdown
- **Delete** a room with the red button at the bottom

### Scale and Measurements

- If the floorplan includes a scale notation (e.g. "1:200"), it is detected automatically
- All measurements show both **real-world units** (m, m²) and **pixel values** (px, px²)
- Wall segments smaller than 3% of the perimeter are hidden (shown as "tiny segments hidden")
- The scale ratio and conversion factors are displayed in the detail panel

### Managing Projects

- Previously processed floorplans appear in the **Previous Projects** list on the home screen
- Click a project to reload it instantly (no reprocessing needed)
- Each project shows: name, timestamp, room count, and scale
- Duplicate filenames trigger a warning before reprocessing
- Delete individual projects with the **x** button, or use checkboxes for **bulk delete**

### Exporting Data

Click the **Export** button at the bottom of the room sidebar to download an Excel file with all room data.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI, uvicorn |
| Computer Vision | OpenCV, Shapely |
| AI Vision | Google Gemini 2.5 Flash |
| PDF Parsing | PyMuPDF (fitz) |
| Database | SQLite |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Canvas | Fabric.js v6 |
| Desktop (optional) | Electron |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process` | Upload and process a floorplan |
| GET | `/api/progress/{job_id}` | Poll processing progress |
| GET | `/api/projects` | List all projects |
| GET | `/api/projects/{id}` | Get project details |
| DELETE | `/api/projects/{id}` | Delete a project |
| GET | `/api/projects/{id}/rooms` | List rooms in a project |
| POST | `/api/rooms` | Create a manual room |
| PUT | `/api/rooms/{id}` | Update a room (name, type, or polygon) |
| DELETE | `/api/rooms/{id}` | Delete a room |
| PUT | `/api/projects/{id}/scale` | Update scale (recalculates all rooms) |
| GET | `/api/projects/{id}/image` | Get the floorplan image |
| GET | `/api/export/{id}?format=xlsx` | Export project data |

## Free Tier Limits (Gemini)

Each floorplan processing run makes **2 Gemini API calls**. On the free tier (Gemini 2.5 Flash):

- 5 requests per minute — allows ~2 floorplans per minute
- ~394K tokens per minute — plenty for large images
- No daily cap — process as many as you want, just not too fast
