import { useFloorplanCanvas } from '../hooks/useFloorplanCanvas';
import type { Room } from '../api';

interface FloorplanCanvasProps {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
}

export function FloorplanCanvas(props: FloorplanCanvasProps) {
  const { containerRef, canvasRef, fitToView } = useFloorplanCanvas(props);

  return (
    <div ref={containerRef} className="relative flex-1 min-w-0 bg-neutral-900 overflow-hidden">
      <div className="absolute top-3 left-3 z-10 flex gap-1">
        <button
          onClick={fitToView}
          className="px-3 py-1.5 bg-neutral-800 text-neutral-300 text-xs rounded hover:bg-neutral-700"
        >
          Fit View
        </button>
      </div>
      <canvas ref={canvasRef} />
      <div className="absolute bottom-3 left-3 text-neutral-600 text-xs">
        Scroll to zoom · Alt+Drag to pan · Click room to select
      </div>
    </div>
  );
}
