import { useEffect, useRef, useCallback, useState } from 'react';
import { Canvas, FabricImage, Polygon, FabricText, Circle, Line, Point } from 'fabric';
import type { Room } from '../api';
import type { CanvasMode } from '../components/FloorplanCanvas';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

interface UseFloorplanCanvasOptions {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  canvasMode: CanvasMode;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
  onRoomDrawn: (polygon: number[][]) => void;
}

export function useFloorplanCanvas({
  rooms, imageUrl, selectedRoomId, canvasMode,
  onRoomSelect, onRoomUpdate, onRoomDrawn,
}: UseFloorplanCanvasOptions) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<Canvas | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Drawing state refs (persist across renders without triggering re-renders)
  const drawVerticesRef = useRef<number[][]>([]);
  const [drawVertexCount, setDrawVertexCount] = useState(0);
  const drawObjectsRef = useRef<any[]>([]);
  const previewLineRef = useRef<Line | null>(null);

  // Edit state refs
  const editHandlesRef = useRef<Circle[]>([]);
  const midpointHandlesRef = useRef<Circle[]>([]);
  const draggingHandleRef = useRef<number>(-1);
  const isMouseDownRef = useRef(false);

  // Edge drag state: when dragging an edge midpoint, both endpoints move
  const draggingEdgeRef = useRef<{ edgeIndex: number; startX: number; startY: number } | null>(null);
  const edgeStartPositionsRef = useRef<{ v1: [number, number]; v2: [number, number] } | null>(null);

  // Ghost polygon preview during drag
  const ghostPolygonRef = useRef<Polygon | null>(null);

  // Undo stack: stores polygon snapshots before each edit
  const undoStackRef = useRef<number[][][]>([]);
  const undoRoomIdRef = useRef<string | null>(null);

  // Store latest callback refs to avoid stale closures
  const onRoomSelectRef = useRef(onRoomSelect);
  const onRoomUpdateRef = useRef(onRoomUpdate);
  const onRoomDrawnRef = useRef(onRoomDrawn);
  const canvasModeRef = useRef(canvasMode);
  const selectedRoomIdRef = useRef(selectedRoomId);
  const roomsRef = useRef(rooms);

  useEffect(() => { onRoomSelectRef.current = onRoomSelect; }, [onRoomSelect]);
  useEffect(() => { onRoomUpdateRef.current = onRoomUpdate; }, [onRoomUpdate]);
  useEffect(() => { onRoomDrawnRef.current = onRoomDrawn; }, [onRoomDrawn]);
  useEffect(() => { canvasModeRef.current = canvasMode; }, [canvasMode]);
  useEffect(() => { selectedRoomIdRef.current = selectedRoomId; }, [selectedRoomId]);
  useEffect(() => { roomsRef.current = rooms; }, [rooms]);

  // --- Canvas init (runs once) ---
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;
    const container = containerRef.current;
    const canvas = new Canvas(canvasRef.current, {
      selection: false,
      backgroundColor: '#111',
      width: container.clientWidth,
      height: container.clientHeight,
    });

    const resizeObserver = new ResizeObserver(() => {
      canvas.setDimensions({
        width: container.clientWidth,
        height: container.clientHeight,
      });
      canvas.requestRenderAll();
    });
    resizeObserver.observe(container);

    let isPanning = false;
    let lastPosX = 0;
    let lastPosY = 0;

    canvas.on('mouse:down', (e) => {
      const evt = e.e as MouseEvent;
      const mode = canvasModeRef.current;

      // Pan: Alt+click or middle mouse in any mode
      if (evt.altKey || evt.button === 1) {
        isPanning = true;
        lastPosX = evt.clientX;
        lastPosY = evt.clientY;
        canvas.setCursor('grabbing');
        return;
      }

      if (mode === 'draw') {
        _handleDrawClick(canvas, e);
        return;
      }

      if (mode === 'edit') {
        _handleEditMouseDown(canvas, e);
        return;
      }

      // View mode: room selection handled by polygon mousedown events
    });

    canvas.on('mouse:dblclick', (e) => {
      if (canvasModeRef.current === 'draw') {
        _finishDraw(canvas);
      }
    });

    canvas.on('mouse:move', (e) => {
      const evt = e.e as MouseEvent;

      if (isPanning) {
        const vpt = canvas.viewportTransform;
        vpt[4] += evt.clientX - lastPosX;
        vpt[5] += evt.clientY - lastPosY;
        lastPosX = evt.clientX;
        lastPosY = evt.clientY;
        canvas.requestRenderAll();
        return;
      }

      if (canvasModeRef.current === 'draw') {
        _handleDrawMouseMove(canvas, e);
        return;
      }

      if (canvasModeRef.current === 'edit' && (draggingHandleRef.current >= 0 || draggingEdgeRef.current)) {
        _handleEditMouseMove(canvas, e);
        return;
      }
    });

    canvas.on('mouse:up', () => {
      isPanning = false;
      isMouseDownRef.current = false;
      canvas.setCursor('default');

      if (canvasModeRef.current === 'edit' && (draggingHandleRef.current >= 0 || draggingEdgeRef.current)) {
        _handleEditMouseUp(canvas);
      }
    });

    canvas.on('mouse:wheel', (opt) => {
      const delta = (opt.e as WheelEvent).deltaY;
      let zoom = canvas.getZoom();
      zoom *= 0.999 ** delta;
      zoom = Math.min(Math.max(0.1, zoom), 20);
      canvas.zoomToPoint(new Point((opt.e as WheelEvent).offsetX, (opt.e as WheelEvent).offsetY), zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });

    fabricRef.current = canvas;
    setIsReady(true);

    // Keyboard handler
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (canvasModeRef.current === 'draw') {
          _cancelDraw(canvas);
        }
      }
      if (e.key === 'Enter' && canvasModeRef.current === 'draw') {
        _finishDraw(canvas);
      }
      // Undo: Ctrl+Z (Windows/Linux) or Cmd+Z (Mac)
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        if (canvasModeRef.current === 'edit') {
          e.preventDefault();
          _handleUndo();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      resizeObserver.disconnect();
      canvas.dispose();
      fabricRef.current = null;
    };
  }, []);

  // --- Helper: convert screen coords to canvas (image) coords ---
  function _screenToCanvas(canvas: Canvas, e: any): { x: number; y: number } {
    const pointer = canvas.getScenePoint(e.e);
    return { x: pointer.x, y: pointer.y };
  }

  // --- Draw mode handlers ---
  function _handleDrawClick(canvas: Canvas, e: any) {
    const pt = _screenToCanvas(canvas, e);
    const vertices = drawVerticesRef.current;
    vertices.push([pt.x, pt.y]);
    setDrawVertexCount(vertices.length);

    // Add vertex marker
    const marker = new Circle({
      left: pt.x - 4,
      top: pt.y - 4,
      radius: 4,
      fill: '#10b981',
      stroke: '#fff',
      strokeWidth: 1,
      selectable: false,
      evented: false,
      data: { type: 'draw' },
    });
    canvas.add(marker);
    drawObjectsRef.current.push(marker);

    // Add line from previous vertex
    if (vertices.length > 1) {
      const prev = vertices[vertices.length - 2];
      const line = new Line([prev[0], prev[1], pt.x, pt.y], {
        stroke: '#10b981',
        strokeWidth: 2,
        selectable: false,
        evented: false,
        data: { type: 'draw' },
      });
      canvas.add(line);
      drawObjectsRef.current.push(line);
    }

    canvas.requestRenderAll();
  }

  function _handleDrawMouseMove(canvas: Canvas, e: any) {
    const vertices = drawVerticesRef.current;
    if (vertices.length === 0) return;

    const pt = _screenToCanvas(canvas, e);
    const last = vertices[vertices.length - 1];

    // Remove old preview line
    if (previewLineRef.current) {
      canvas.remove(previewLineRef.current);
    }

    const line = new Line([last[0], last[1], pt.x, pt.y], {
      stroke: '#10b981',
      strokeWidth: 1,
      strokeDashArray: [6, 3],
      selectable: false,
      evented: false,
      data: { type: 'draw' },
    });
    canvas.add(line);
    previewLineRef.current = line;
    canvas.requestRenderAll();
  }

  function _finishDraw(canvas: Canvas) {
    const vertices = drawVerticesRef.current;
    if (vertices.length < 3) return;

    const polygon = [...vertices];
    _clearDrawObjects(canvas);
    onRoomDrawnRef.current(polygon);
  }

  function _cancelDraw(canvas: Canvas) {
    _clearDrawObjects(canvas);
  }

  function _clearDrawObjects(canvas: Canvas) {
    for (const obj of drawObjectsRef.current) {
      canvas.remove(obj);
    }
    if (previewLineRef.current) {
      canvas.remove(previewLineRef.current);
      previewLineRef.current = null;
    }
    drawObjectsRef.current = [];
    drawVerticesRef.current = [];
    setDrawVertexCount(0);
    canvas.requestRenderAll();
  }

  // --- Undo helpers ---
  function _saveUndoState() {
    const roomId = selectedRoomIdRef.current;
    if (!roomId) return;
    // Reset stack if switching rooms
    if (undoRoomIdRef.current !== roomId) {
      undoStackRef.current = [];
      undoRoomIdRef.current = roomId;
    }
    // Capture current polygon from handles
    const snapshot: number[][] = editHandlesRef.current.map((h) => [
      (h.left ?? 0) + 5, (h.top ?? 0) + 5,
    ]);
    undoStackRef.current.push(snapshot);
  }

  function _handleUndo() {
    const stack = undoStackRef.current;
    if (stack.length === 0) return;
    const prev = stack.pop()!;
    const roomId = selectedRoomIdRef.current;
    if (roomId) {
      onRoomUpdateRef.current(roomId, prev);
    }
  }

  // --- Edit mode handlers ---
  function _handleEditMouseDown(canvas: Canvas, e: any) {
    isMouseDownRef.current = true;
    const target = canvas.findTarget(e.e);

    // Vertex handle drag
    if (target && (target as any).data?.type === 'editHandle') {
      _saveUndoState();
      draggingHandleRef.current = (target as any).data.vertexIndex;
      canvas.setCursor('move');
      return;
    }

    // Edge midpoint handle drag — translate entire edge in parallel
    if (target && (target as any).data?.type === 'edgeMidpoint') {
      _saveUndoState();
      const edgeIdx: number = (target as any).data.edgeIndex;
      const pt = _screenToCanvas(canvas, e);
      const handles = editHandlesRef.current;
      const nextIdx = (edgeIdx + 1) % handles.length;

      draggingEdgeRef.current = { edgeIndex: edgeIdx, startX: pt.x, startY: pt.y };
      edgeStartPositionsRef.current = {
        v1: [(handles[edgeIdx].left ?? 0) + 5, (handles[edgeIdx].top ?? 0) + 5],
        v2: [(handles[nextIdx].left ?? 0) + 5, (handles[nextIdx].top ?? 0) + 5],
      };
      canvas.setCursor('move');
      return;
    }

    // Click on room polygon — select it
    if (target && (target as any).data?.type === 'room') {
      onRoomSelectRef.current((target as any).data.roomId);
      return;
    }

    // Clicked empty space — deselect
    onRoomSelectRef.current(null);
  }

  function _buildGhostPoints(): { x: number; y: number }[] {
    return editHandlesRef.current.map((h) => ({
      x: (h.left ?? 0) + 5,
      y: (h.top ?? 0) + 5,
    }));
  }

  function _updateGhostPolygon(canvas: Canvas) {
    // Remove old ghost
    if (ghostPolygonRef.current) {
      canvas.remove(ghostPolygonRef.current);
      ghostPolygonRef.current = null;
    }
    // Draw new ghost showing the in-progress shape
    const pts = _buildGhostPoints();
    if (pts.length < 3) return;
    const ghost = new Polygon(pts, {
      fill: '#facc1510',
      stroke: '#facc15',
      strokeWidth: 2,
      strokeDashArray: [6, 4],
      selectable: false,
      evented: false,
      data: { type: 'ghost' },
    });
    canvas.add(ghost);
    // Move ghost below handles so handles remain clickable
    canvas.sendObjectToBack(ghost);
    ghostPolygonRef.current = ghost;
  }

  function _handleEditMouseMove(canvas: Canvas, e: any) {
    // Only drag while mouse button is held
    if (!isMouseDownRef.current) return;

    const pt = _screenToCanvas(canvas, e);

    // Edge drag: translate both endpoints by the same delta
    if (draggingEdgeRef.current && edgeStartPositionsRef.current) {
      const { edgeIndex, startX, startY } = draggingEdgeRef.current;
      const { v1, v2 } = edgeStartPositionsRef.current;
      const dx = pt.x - startX;
      const dy = pt.y - startY;
      const handles = editHandlesRef.current;
      const nextIdx = (edgeIndex + 1) % handles.length;

      handles[edgeIndex].set({ left: v1[0] + dx - 5, top: v1[1] + dy - 5 });
      handles[edgeIndex].setCoords();
      handles[nextIdx].set({ left: v2[0] + dx - 5, top: v2[1] + dy - 5 });
      handles[nextIdx].setCoords();
      _updateGhostPolygon(canvas);
      canvas.requestRenderAll();
      return;
    }

    // Vertex drag
    const idx = draggingHandleRef.current;
    if (idx < 0) return;
    const handle = editHandlesRef.current[idx];
    if (!handle) return;

    handle.set({ left: pt.x - 5, top: pt.y - 5 });
    handle.setCoords();
    _updateGhostPolygon(canvas);
    canvas.requestRenderAll();
  }

  function _handleEditMouseUp(canvas: Canvas) {
    isMouseDownRef.current = false;
    const isEdgeDrag = draggingEdgeRef.current !== null;
    const isVertexDrag = draggingHandleRef.current >= 0;

    // Clean up ghost polygon
    if (ghostPolygonRef.current) {
      canvas.remove(ghostPolygonRef.current);
      ghostPolygonRef.current = null;
    }

    if (!isEdgeDrag && !isVertexDrag) return;

    draggingHandleRef.current = -1;
    draggingEdgeRef.current = null;
    edgeStartPositionsRef.current = null;
    canvas.setCursor('default');

    // Build updated polygon from handle positions
    const newPolygon: number[][] = editHandlesRef.current.map((h) => [
      (h.left ?? 0) + 5, (h.top ?? 0) + 5,
    ]);

    const roomId = selectedRoomIdRef.current;
    if (roomId) {
      onRoomUpdateRef.current(roomId, newPolygon);
    }
  }

  // --- Load background image ---
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !imageUrl) return;

    FabricImage.fromURL(imageUrl, { crossOrigin: 'anonymous' }).then((img) => {
      if (!img || !img.width) return;
      img.set({ selectable: false, evented: false });
      canvas.backgroundImage = img;

      const cw = canvas.width ?? 800;
      const ch = canvas.height ?? 600;
      const iw = img.width || 1;
      const ih = img.height || 1;
      const scale = Math.min(cw / iw, ch / ih) * 0.95;
      const offsetX = (cw - iw * scale) / 2;
      const offsetY = (ch - ih * scale) / 2;
      canvas.viewportTransform = [scale, 0, 0, scale, offsetX, offsetY];
      canvas.requestRenderAll();
    }).catch((err) => console.error('Image load error:', err));
  }, [imageUrl, isReady]);

  // --- Render room polygons + labels ---
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    // Remove old room objects and ghost polygon
    const objects = canvas.getObjects().filter((o) => {
      const t = (o as any).data?.type;
      return t === 'room' || t === 'editHandle' || t === 'edgeMidpoint' || t === 'ghost';
    });
    objects.forEach((o) => canvas.remove(o));
    editHandlesRef.current = [];
    midpointHandlesRef.current = [];
    ghostPolygonRef.current = null;

    rooms.forEach((room, i) => {
      if (!room.boundary_polygon || room.boundary_polygon.length < 3) return;
      const points = room.boundary_polygon.map((p) => ({ x: p[0], y: p[1] }));
      const color = ROOM_COLORS[i % ROOM_COLORS.length];
      const isSelected = room.id === selectedRoomId;

      const polygon = new Polygon(points, {
        fill: isSelected ? '#facc1550' : color + '20',
        stroke: isSelected ? '#facc15' : color,
        strokeWidth: isSelected ? 3 : 1,
        strokeDashArray: isSelected ? [8, 4] : undefined,
        selectable: false,
        data: { type: 'room', roomId: room.id },
      });

      // In view or edit mode, clicking a polygon selects it
      if (canvasMode === 'view' || canvasMode === 'edit') {
        polygon.on('mousedown', () => onRoomSelect(room.id));
      }
      canvas.add(polygon);

      const label = new FabricText(room.name || 'Unnamed', {
        left: room.centroid[0],
        top: room.centroid[1],
        fontSize: isSelected ? 14 : 12,
        fill: isSelected ? '#facc15' : color,
        fontWeight: isSelected ? 'bold' : 'normal',
        fontFamily: 'monospace',
        selectable: false,
        evented: false,
        data: { type: 'room' },
      });
      canvas.add(label);

      // Edit mode: show vertex handles on selected room
      if (canvasMode === 'edit' && isSelected) {
        const coords = room.boundary_polygon;
        // Skip last point if it duplicates the first (closed polygon)
        const vertexCount =
          coords.length > 1 &&
          coords[0][0] === coords[coords.length - 1][0] &&
          coords[0][1] === coords[coords.length - 1][1]
            ? coords.length - 1
            : coords.length;

        for (let vi = 0; vi < vertexCount; vi++) {
          const vx = coords[vi][0];
          const vy = coords[vi][1];
          const handle = new Circle({
            left: vx - 5,
            top: vy - 5,
            radius: 5,
            fill: '#facc15',
            stroke: '#000',
            strokeWidth: 1,
            selectable: false,
            evented: true,
            hoverCursor: 'move',
            data: { type: 'editHandle', vertexIndex: vi },
          });
          canvas.add(handle);
          editHandlesRef.current.push(handle);
        }

        // Edge midpoint handles — smaller, semi-transparent
        for (let vi = 0; vi < vertexCount; vi++) {
          const nextVi = (vi + 1) % vertexCount;
          const mx = (coords[vi][0] + coords[nextVi][0]) / 2;
          const my = (coords[vi][1] + coords[nextVi][1]) / 2;
          const mpHandle = new Circle({
            left: mx - 3,
            top: my - 3,
            radius: 3,
            fill: '#facc1580',
            stroke: '#00000080',
            strokeWidth: 1,
            selectable: false,
            evented: true,
            hoverCursor: 'grab',
            data: { type: 'edgeMidpoint', edgeIndex: vi },
          });
          canvas.add(mpHandle);
          midpointHandlesRef.current.push(mpHandle);
        }
      }
    });
    canvas.requestRenderAll();
  }, [rooms, selectedRoomId, canvasMode, isReady, onRoomSelect]);

  // Clear draw objects when mode changes away from draw
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    if (canvasMode !== 'draw') {
      _clearDrawObjects(canvas);
    }
  }, [canvasMode]);

  const fitToView = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas || !canvas.backgroundImage) return;
    const img = canvas.backgroundImage as FabricImage;
    const cw = canvas.width ?? 800;
    const ch = canvas.height ?? 600;
    const iw = img.width || 1;
    const ih = img.height || 1;
    const scale = Math.min(cw / iw, ch / ih) * 0.95;
    const offsetX = (cw - iw * scale) / 2;
    const offsetY = (ch - ih * scale) / 2;
    canvas.viewportTransform = [scale, 0, 0, scale, offsetX, offsetY];
    canvas.requestRenderAll();
  }, []);

  return { containerRef, canvasRef, fitToView, drawVertexCount };
}
