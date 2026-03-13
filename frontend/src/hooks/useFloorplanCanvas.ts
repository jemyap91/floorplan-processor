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
  const draggingHandleRef = useRef<number>(-1);

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

      if (canvasModeRef.current === 'edit' && draggingHandleRef.current >= 0) {
        _handleEditMouseMove(canvas, e);
        return;
      }
    });

    canvas.on('mouse:up', () => {
      isPanning = false;
      canvas.setCursor('default');

      if (canvasModeRef.current === 'edit' && draggingHandleRef.current >= 0) {
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

  // --- Edit mode handlers ---
  function _handleEditMouseDown(canvas: Canvas, e: any) {
    // Check if clicking on a handle
    const target = canvas.findTarget(e.e);
    if (target && (target as any).data?.type === 'editHandle') {
      draggingHandleRef.current = (target as any).data.vertexIndex;
      canvas.setCursor('move');
      return;
    }

    // Check if clicking on a room polygon
    if (target && (target as any).data?.type === 'room') {
      onRoomSelectRef.current((target as any).data.roomId);
      return;
    }

    // Clicked empty space — deselect
    onRoomSelectRef.current(null);
  }

  function _handleEditMouseMove(canvas: Canvas, e: any) {
    const idx = draggingHandleRef.current;
    if (idx < 0) return;

    const pt = _screenToCanvas(canvas, e);
    const handle = editHandlesRef.current[idx];
    if (!handle) return;

    handle.set({ left: pt.x - 5, top: pt.y - 5 });
    handle.setCoords();
    canvas.requestRenderAll();
  }

  function _handleEditMouseUp(canvas: Canvas) {
    const idx = draggingHandleRef.current;
    if (idx < 0) return;
    draggingHandleRef.current = -1;
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

    // Remove old room objects
    const objects = canvas.getObjects().filter((o) => {
      const t = (o as any).data?.type;
      return t === 'room' || t === 'editHandle';
    });
    objects.forEach((o) => canvas.remove(o));
    editHandlesRef.current = [];

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
