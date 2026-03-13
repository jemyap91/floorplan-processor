import { useFloorplanCanvas } from '../hooks/useFloorplanCanvas';
import type { Room } from '../api';

export type CanvasMode = 'view' | 'draw' | 'edit';

interface FloorplanCanvasProps {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  canvasMode: CanvasMode;
  onCanvasModeChange: (mode: CanvasMode) => void;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
  onRoomDrawn: (polygon: number[][]) => void;
}

export function FloorplanCanvas({
  rooms, imageUrl, selectedRoomId, canvasMode, onCanvasModeChange,
  onRoomSelect, onRoomUpdate, onRoomDrawn,
}: FloorplanCanvasProps) {
  const { containerRef, canvasRef, fitToView, drawVertexCount } = useFloorplanCanvas({
    rooms, imageUrl, selectedRoomId, canvasMode,
    onRoomSelect, onRoomUpdate, onRoomDrawn,
  });

  return (
    <div ref={containerRef} className="relative flex-1 min-w-0 bg-neutral-900 overflow-hidden">
      {/* Toolbar */}
      <div className="absolute top-3 left-3 z-10 flex gap-1">
        <button
          onClick={() => onCanvasModeChange('view')}
          className={`px-3 py-1.5 text-xs rounded transition-colors ${
            canvasMode === 'view'
              ? 'bg-sky-600 text-white'
              : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
          }`}
          title="View mode — click rooms to select"
        >
          <span className="mr-1">&#9670;</span> View
        </button>
        <button
          onClick={() => onCanvasModeChange('draw')}
          className={`px-3 py-1.5 text-xs rounded transition-colors ${
            canvasMode === 'draw'
              ? 'bg-emerald-600 text-white'
              : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
          }`}
          title="Draw mode — click to place vertices, double-click to close"
        >
          <span className="mr-1">&#9998;</span> Draw
        </button>
        <button
          onClick={() => onCanvasModeChange('edit')}
          className={`px-3 py-1.5 text-xs rounded transition-colors ${
            canvasMode === 'edit'
              ? 'bg-amber-600 text-white'
              : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
          }`}
          title="Edit mode — drag vertices to reshape rooms"
        >
          <span className="mr-1">&#8982;</span> Edit
        </button>
        <div className="w-px bg-neutral-700 mx-1" />
        <button
          onClick={fitToView}
          className="px-3 py-1.5 bg-neutral-800 text-neutral-300 text-xs rounded hover:bg-neutral-700"
        >
          Fit View
        </button>
      </div>
      <canvas ref={canvasRef} />
      <div className="absolute bottom-3 left-3 text-neutral-600 text-xs">
        {canvasMode === 'view' && 'Scroll to zoom · Alt+Drag to pan · Click room to select'}
        {canvasMode === 'draw' && (
          <>
            Click to place vertices{drawVertexCount > 0 && ` (${drawVertexCount} placed)`} · Double-click to close · Esc to cancel
          </>
        )}
        {canvasMode === 'edit' && 'Click room to select · Drag vertices to reshape · Esc to deselect'}
      </div>
    </div>
  );
}
