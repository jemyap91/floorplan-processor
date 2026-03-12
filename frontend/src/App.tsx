import { useState, useCallback, useEffect } from 'react';
import { FloorplanCanvas } from './components/FloorplanCanvas';
import { RoomSidebar } from './components/RoomSidebar';
import { RoomDetail } from './components/RoomDetail';
import {
  processFloorplan, updateRoom, deleteRoom, getImageUrl,
  getProjects, getRooms,
  type Room, type ProcessMode, type ScaleInfo, type Project,
} from './api';

type AppState = 'idle' | 'processing' | 'ready' | 'error';

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [scale, setScale] = useState<ScaleInfo | null>(null);
  const [progress, setProgress] = useState('');
  const [mode, setMode] = useState<ProcessMode>('gemini');
  const [previousProjects, setPreviousProjects] = useState<Project[]>([]);

  const selectedRoom = rooms.find((r) => r.id === selectedRoomId) || null;
  const imageUrl = projectId ? getImageUrl(projectId) : null;

  // Load previous projects on mount
  useEffect(() => {
    getProjects()
      .then((projects) => setPreviousProjects(projects))
      .catch(() => {});
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    setState('processing');
    setError(null);
    setProgress(mode === 'gemini'
      ? 'Sending to Gemini for room extraction...'
      : 'Processing with CV + Gemini pipeline...',
    );
    try {
      const result = await processFloorplan(file, 0, mode);
      setProjectId(result.project_id);
      setRooms(result.rooms);
      setScale(result.scale);
      setState('ready');
    } catch (err: any) {
      setError(err.message || 'Processing failed');
      setState('error');
    }
  }, [mode]);

  const handleLoadProject = useCallback(async (project: Project) => {
    setState('processing');
    setError(null);
    setProgress('Loading saved project...');
    try {
      const projectRooms = await getRooms(project.id);
      setProjectId(project.id);
      setRooms(projectRooms);
      setScale({
        px_per_meter: project.scale_px_per_meter,
        source: project.scale_source,
      });
      setState('ready');
    } catch (err: any) {
      setError(err.message || 'Failed to load project');
      setState('error');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
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

  const handleBackToHome = useCallback(() => {
    setState('idle');
    setProjectId(null);
    setRooms([]);
    setSelectedRoomId(null);
    setScale(null);
    setError(null);
    // Refresh project list
    getProjects()
      .then((projects) => setPreviousProjects(projects))
      .catch(() => {});
  }, []);

  if (state === 'idle' || state === 'error') {
    return (
      <div
        className="h-screen bg-neutral-950 flex items-center justify-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="text-center space-y-5 max-w-lg">
          <h1 className="text-2xl font-bold text-neutral-200">Floorplan Processor</h1>
          <p className="text-neutral-500">Drop a floorplan (PDF, PNG, JPG) or click to upload</p>
          <div className="flex items-center justify-center gap-3">
            <span className={`text-xs ${mode === 'hybrid' ? 'text-neutral-200' : 'text-neutral-600'}`}>CV + AI</span>
            <button
              onClick={() => setMode(mode === 'hybrid' ? 'gemini' : 'hybrid')}
              className={`relative w-11 h-6 rounded-full transition-colors shrink-0 ${
                mode === 'gemini' ? 'bg-violet-600' : 'bg-neutral-700'
              }`}
            >
              <span className={`block w-4 h-4 rounded-full bg-white absolute top-1 transition-all ${
                mode === 'gemini' ? 'left-6' : 'left-1'
              }`} />
            </button>
            <span className={`text-xs ${mode === 'gemini' ? 'text-neutral-200' : 'text-neutral-600'}`}>Gemini AI</span>
          </div>
          <p className="text-neutral-600 text-xs">
            {mode === 'gemini'
              ? 'CV boundaries + Gemini room labelling (recommended)'
              : 'CV boundaries + basic Gemini labelling'}
          </p>
          <label className="inline-block px-6 py-3 bg-sky-600 text-white rounded-lg cursor-pointer hover:bg-sky-500">
            Select File
            <input
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </label>
          {error && <p className="text-red-400 text-sm">{error}</p>}

          {/* Previous projects */}
          {previousProjects.length > 0 && (
            <div className="mt-6 pt-6 border-t border-neutral-800">
              <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-3">
                Previous Projects
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {previousProjects.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleLoadProject(p)}
                    className="w-full text-left px-3 py-2 bg-neutral-900 border border-neutral-800 rounded hover:border-neutral-600 transition-colors"
                  >
                    <div className="text-neutral-200 text-sm truncate">{p.name}</div>
                    <div className="text-neutral-600 text-xs">
                      {new Date(p.created_at).toLocaleDateString()}
                      {p.scale_px_per_meter && ` · ${p.scale_px_per_meter.toFixed(1)} px/m`}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
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
        projectId={projectId}
        onBack={handleBackToHome}
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
        scale={scale}
        onUpdate={handleRoomUpdate}
        onDelete={handleRoomDelete}
      />
    </div>
  );
}
