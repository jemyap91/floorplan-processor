import { useEffect, useRef, useCallback, useState } from 'react';
import { Canvas, FabricImage, Polygon, FabricText, Point } from 'fabric';
import type { Room } from '../api';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

interface UseFloorplanCanvasOptions {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
}

export function useFloorplanCanvas({
  rooms, imageUrl, selectedRoomId, onRoomSelect, onRoomUpdate,
}: UseFloorplanCanvasOptions) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<Canvas | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = new Canvas(canvasRef.current, {
      selection: false,
      backgroundColor: '#111',
    });

    let isPanning = false;
    let lastPosX = 0;
    let lastPosY = 0;

    canvas.on('mouse:down', (e) => {
      if (e.e.altKey || (e.e as MouseEvent).button === 1) {
        isPanning = true;
        lastPosX = (e.e as MouseEvent).clientX;
        lastPosY = (e.e as MouseEvent).clientY;
        canvas.setCursor('grabbing');
      }
    });

    canvas.on('mouse:move', (e) => {
      if (!isPanning) return;
      const vpt = canvas.viewportTransform;
      vpt[4] += (e.e as MouseEvent).clientX - lastPosX;
      vpt[5] += (e.e as MouseEvent).clientY - lastPosY;
      lastPosX = (e.e as MouseEvent).clientX;
      lastPosY = (e.e as MouseEvent).clientY;
      canvas.requestRenderAll();
    });

    canvas.on('mouse:up', () => {
      isPanning = false;
      canvas.setCursor('default');
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

    return () => {
      canvas.dispose();
      fabricRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !imageUrl) return;

    FabricImage.fromURL(imageUrl).then((img) => {
      canvas.setDimensions({
        width: canvas.getElement().parentElement?.clientWidth || 800,
        height: canvas.getElement().parentElement?.clientHeight || 600,
      });
      img.set({ selectable: false, evented: false });
      canvas.backgroundImage = img;
      const scaleX = (canvas.width ?? 800) / (img.width || 1);
      const scaleY = (canvas.height ?? 600) / (img.height || 1);
      const scale = Math.min(scaleX, scaleY);
      canvas.setZoom(scale);
      canvas.requestRenderAll();
    });
  }, [imageUrl, isReady]);

  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    const objects = canvas.getObjects().filter((o) => (o as any).data?.type === 'room');
    objects.forEach((o) => canvas.remove(o));

    rooms.forEach((room, i) => {
      if (!room.boundary_polygon || room.boundary_polygon.length < 3) return;
      const points = room.boundary_polygon.map((p) => ({ x: p[0], y: p[1] }));
      const color = ROOM_COLORS[i % ROOM_COLORS.length];
      const isSelected = room.id === selectedRoomId;

      const polygon = new Polygon(points, {
        fill: isSelected ? color + '40' : color + '20',
        stroke: color,
        strokeWidth: isSelected ? 3 : 1.5,
        selectable: false,
        data: { type: 'room', roomId: room.id },
      });
      polygon.on('mousedown', () => onRoomSelect(room.id));
      canvas.add(polygon);

      const label = new FabricText(room.name || 'Unnamed', {
        left: room.centroid[0],
        top: room.centroid[1],
        fontSize: 12,
        fill: color,
        fontFamily: 'monospace',
        selectable: false,
        evented: false,
        data: { type: 'room' },
      });
      canvas.add(label);
    });
    canvas.requestRenderAll();
  }, [rooms, selectedRoomId, isReady, onRoomSelect]);

  const fitToView = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas || !canvas.backgroundImage) return;
    const img = canvas.backgroundImage as FabricImage;
    const scaleX = (canvas.width ?? 800) / (img.width || 1);
    const scaleY = (canvas.height ?? 600) / (img.height || 1);
    canvas.setZoom(Math.min(scaleX, scaleY));
    canvas.viewportTransform = [canvas.getZoom(), 0, 0, canvas.getZoom(), 0, 0];
    canvas.requestRenderAll();
  }, []);

  return { canvasRef, fitToView };
}
