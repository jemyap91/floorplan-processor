import { useState, useCallback } from 'react';
import { FloorplanCanvas } from './components/FloorplanCanvas';
import { RoomSidebar } from './components/RoomSidebar';
import { RoomDetail } from './components/RoomDetail';
import {
  processFloorplan, updateRoom, deleteRoom, getImageUrl,
  type Room, type ProcessResult,
} from './api';

type AppState = 'idle' | 'processing' | 'ready' | 'error';

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [scale, setScale] = useState<ProcessResult['scale'] | null>(null);
  const [progress, setProgress] = useState('');

  const selectedRoom = rooms.find((r) => r.id === selectedRoomId) || null;
  const imageUrl = projectId ? getImageUrl(projectId) : null;

  const handleFileUpload = useCallback(async (file: File) => {
    setState('processing');
    setError(null);
    setProgress('Uploading and processing...');
    try {
      const result = await processFloorplan(file);
      setProjectId(result.project_id);
      setRooms(result.rooms);
      setScale(result.scale);
      setState('ready');
    } catch (err: any) {
      setError(err.message || 'Processing failed');
      setState('error');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file?.type === 'application/pdf') handleFileUpload(file);
  }, [handleFileUpload]);

  const handleRoomUpdate = useCallback(async (roomId: string, update: Partial<Room>) => {
    await updateRoom(roomId, update);
    setRooms((prev) => prev.map((r) => r.id === roomId ? { ...r, ...update } : r));
  }, []);

  const handleRoomDelete = useCallback(async (roomId: string) => {
    await deleteRoom(roomId);
    setRooms((prev) => prev.filter((r) => r.id !== roomId));
    if (selectedRoomId === roomId) setSelectedRoomId(null);
  }, [selectedRoomId]);

  const handleRoomPolygonUpdate = useCallback((roomId: string, polygon: number[][]) => {
    handleRoomUpdate(roomId, { boundary_polygon: polygon });
  }, [handleRoomUpdate]);

  if (state === 'idle' || state === 'error') {
    return (
      <div
        className="h-screen bg-neutral-950 flex items-center justify-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-bold text-neutral-200">Floorplan Processor</h1>
          <p className="text-neutral-500">Drop a PDF floorplan or click to upload</p>
          <label className="inline-block px-6 py-3 bg-sky-600 text-white rounded-lg cursor-pointer hover:bg-sky-500">
            Select PDF
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </label>
          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      </div>
    );
  }

  if (state === 'processing') {
    return (
      <div className="h-screen bg-neutral-950 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full mx-auto" />
          <p className="text-neutral-400">{progress}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-neutral-950 flex overflow-hidden">
      <RoomSidebar
        rooms={rooms}
        selectedRoomId={selectedRoomId}
        onRoomSelect={setSelectedRoomId}
        scale={scale}
      />
      <FloorplanCanvas
        rooms={rooms}
        imageUrl={imageUrl}
        selectedRoomId={selectedRoomId}
        onRoomSelect={setSelectedRoomId}
        onRoomUpdate={handleRoomPolygonUpdate}
      />
      <RoomDetail
        room={selectedRoom}
        onUpdate={handleRoomUpdate}
        onDelete={handleRoomDelete}
      />
    </div>
  );
}
