import { useState, useCallback, useEffect, useRef } from 'react';
import { FloorplanCanvas } from './components/FloorplanCanvas';
import { RoomSidebar } from './components/RoomSidebar';
import { RoomDetail } from './components/RoomDetail';
import {
  processFloorplan, updateRoom, deleteRoom, getImageUrl,
  getProjects, getRooms, deleteProject, getProgress, createRoom,
  type Room, type ProcessMode, type ScaleInfo, type Project,
} from './api';
import type { CanvasMode } from './components/FloorplanCanvas';

type AppState = 'idle' | 'processing' | 'ready' | 'error';

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
    + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

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
  const [duplicateWarning, setDuplicateWarning] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [selectedProjectIds, setSelectedProjectIds] = useState<Set<string>>(new Set());
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false);
  const [percent, setPercent] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [canvasMode, setCanvasMode] = useState<CanvasMode>('view');
  const [newRoomPolygon, setNewRoomPolygon] = useState<number[][] | null>(null);
  const [newRoomName, setNewRoomName] = useState('');
  const [newRoomType, setNewRoomType] = useState('unknown');

  const selectedRoom = rooms.find((r) => r.id === selectedRoomId) || null;
  const imageUrl = projectId ? getImageUrl(projectId) : null;

  const refreshProjects = useCallback(() => {
    getProjects()
      .then((projects) => setPreviousProjects(projects))
      .catch(() => {});
  }, []);

  useEffect(() => { refreshProjects(); }, [refreshProjects]);

  const doUpload = useCallback(async (file: File) => {
    setState('processing');
    setError(null);
    setDuplicateWarning(null);
    setPendingFile(null);
    setPercent(0);
    setProgress('Starting...');

    const jobId = crypto.randomUUID();

    // Poll progress every 500ms
    pollRef.current = setInterval(async () => {
      try {
        const p = await getProgress(jobId);
        setPercent(p.percent);
        setProgress(p.message);
      } catch { /* ignore */ }
    }, 500);

    try {
      const result = await processFloorplan(file, 0, mode, jobId);
      setProjectId(result.project_id);
      setRooms(result.rooms);
      setScale(result.scale);
      setPercent(100);
      setState('ready');
    } catch (err: any) {
      setError(err.message || 'Processing failed');
      setState('error');
    } finally {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [mode]);

  const handleFileUpload = useCallback((file: File) => {
    // Check for duplicate filename
    const baseName = file.name.replace(/\.[^.]+$/, '');
    const matches = previousProjects.filter((p) =>
      p.name.toLowerCase() === baseName.toLowerCase()
    );
    if (matches.length > 0) {
      const latest = matches[0];
      setDuplicateWarning(
        `"${baseName}" was already processed on ${formatTimestamp(latest.created_at)} ` +
        `(${latest.room_count} rooms). Process again?`
      );
      setPendingFile(file);
      return;
    }
    doUpload(file);
  }, [previousProjects, doUpload]);

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

  const handleDeleteProject = useCallback(async (projectId: string) => {
    try {
      await deleteProject(projectId);
      setPreviousProjects((prev) => prev.filter((p) => p.id !== projectId));
      setSelectedProjectIds((prev) => { const next = new Set(prev); next.delete(projectId); return next; });
      setDeleteConfirm(null);
    } catch (err: any) {
      setError(err.message || 'Failed to delete project');
    }
  }, []);

  const handleBulkDelete = useCallback(async () => {
    try {
      await Promise.all([...selectedProjectIds].map((id) => deleteProject(id)));
      setPreviousProjects((prev) => prev.filter((p) => !selectedProjectIds.has(p.id)));
      setSelectedProjectIds(new Set());
      setBulkDeleteConfirm(false);
    } catch (err: any) {
      setError(err.message || 'Failed to delete projects');
    }
  }, [selectedProjectIds]);

  const toggleProjectSelection = useCallback((id: string) => {
    setSelectedProjectIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedProjectIds((prev) =>
      prev.size === previousProjects.length
        ? new Set()
        : new Set(previousProjects.map((p) => p.id))
    );
  }, [previousProjects]);

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

  const handleRoomPolygonUpdate = useCallback(async (roomId: string, polygon: number[][]) => {
    try {
      const updated = await updateRoom(roomId, { polygon });
      // Refresh rooms from server to get recalculated geometry
      if (projectId) {
        const freshRooms = await getRooms(projectId);
        setRooms(freshRooms);
      }
    } catch (err: any) {
      console.error('Failed to update polygon:', err);
    }
  }, [projectId]);

  const handleRoomDrawn = useCallback((polygon: number[][]) => {
    setNewRoomPolygon(polygon);
    setNewRoomName('');
    setNewRoomType('unknown');
    setCanvasMode('view');
  }, []);

  const handleCreateRoom = useCallback(async () => {
    if (!projectId || !newRoomPolygon) return;
    try {
      const room = await createRoom(projectId, newRoomName || 'Unnamed', newRoomType, newRoomPolygon);
      setRooms((prev) => [...prev, room]);
      setNewRoomPolygon(null);
      setSelectedRoomId(room.id);
    } catch (err: any) {
      console.error('Failed to create room:', err);
    }
  }, [projectId, newRoomPolygon, newRoomName, newRoomType]);

  const handleCancelCreateRoom = useCallback(() => {
    setNewRoomPolygon(null);
  }, []);

  const handleBackToHome = useCallback(() => {
    setState('idle');
    setProjectId(null);
    setRooms([]);
    setSelectedRoomId(null);
    setScale(null);
    setError(null);
    setDuplicateWarning(null);
    setPendingFile(null);
    setDeleteConfirm(null);
    setSelectedProjectIds(new Set());
    setBulkDeleteConfirm(false);
    refreshProjects();
  }, [refreshProjects]);

  if (state === 'idle' || state === 'error') {
    const projectToDelete = deleteConfirm
      ? previousProjects.find((p) => p.id === deleteConfirm)
      : null;

    return (
      <div
        className="h-screen bg-neutral-950 flex items-center justify-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="text-center space-y-5 max-w-lg w-full px-4">
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

          {/* Duplicate file warning */}
          {duplicateWarning && pendingFile && (
            <div className="bg-amber-900/30 border border-amber-700 rounded-lg p-3 text-left">
              <p className="text-amber-400 text-sm mb-2">{duplicateWarning}</p>
              <div className="flex gap-2">
                <button
                  onClick={() => doUpload(pendingFile)}
                  className="px-3 py-1 bg-amber-600 text-white text-xs rounded hover:bg-amber-500"
                >
                  Process Again
                </button>
                <button
                  onClick={() => { setDuplicateWarning(null); setPendingFile(null); }}
                  className="px-3 py-1 bg-neutral-700 text-neutral-300 text-xs rounded hover:bg-neutral-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Delete confirmation modal */}
          {deleteConfirm && projectToDelete && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-left">
              <p className="text-red-400 text-sm mb-1">
                Delete "{projectToDelete.name}"?
              </p>
              <p className="text-neutral-500 text-xs mb-2">
                This will permanently remove the project, all {projectToDelete.room_count} rooms, and the stored image. This cannot be undone.
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => handleDeleteProject(deleteConfirm)}
                  className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-500"
                >
                  Delete
                </button>
                <button
                  onClick={() => setDeleteConfirm(null)}
                  className="px-3 py-1 bg-neutral-700 text-neutral-300 text-xs rounded hover:bg-neutral-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Previous projects */}
          {previousProjects.length > 0 && (
            <div className="mt-6 pt-6 border-t border-neutral-800">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-wider">
                  Previous Projects ({previousProjects.length})
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={toggleSelectAll}
                    className="text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
                  >
                    {selectedProjectIds.size === previousProjects.length ? 'Deselect all' : 'Select all'}
                  </button>
                  {selectedProjectIds.size > 0 && (
                    <button
                      onClick={() => setBulkDeleteConfirm(true)}
                      className="text-xs px-2 py-0.5 bg-red-900/40 text-red-400 rounded hover:bg-red-900/60 transition-colors"
                    >
                      Delete {selectedProjectIds.size} selected
                    </button>
                  )}
                </div>
              </div>

              {/* Bulk delete confirmation */}
              {bulkDeleteConfirm && (
                <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-left mb-3">
                  <p className="text-red-400 text-sm mb-1">
                    Delete {selectedProjectIds.size} project{selectedProjectIds.size !== 1 ? 's' : ''}?
                  </p>
                  <p className="text-neutral-500 text-xs mb-2">
                    This will permanently remove {selectedProjectIds.size} project{selectedProjectIds.size !== 1 ? 's' : ''}, all their rooms, and stored images. This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={handleBulkDelete}
                      className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-500"
                    >
                      Delete All Selected
                    </button>
                    <button
                      onClick={() => setBulkDeleteConfirm(false)}
                      className="px-3 py-1 bg-neutral-700 text-neutral-300 text-xs rounded hover:bg-neutral-600"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {previousProjects.map((p) => (
                  <div
                    key={p.id}
                    className={`flex items-center gap-2 bg-neutral-900 border rounded hover:border-neutral-600 transition-colors ${
                      selectedProjectIds.has(p.id) ? 'border-sky-700' : 'border-neutral-800'
                    }`}
                  >
                    <label className="pl-3 flex items-center shrink-0 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedProjectIds.has(p.id)}
                        onChange={() => toggleProjectSelection(p.id)}
                        className="w-3.5 h-3.5 rounded border-neutral-600 bg-neutral-800 accent-sky-500"
                      />
                    </label>
                    <button
                      onClick={() => handleLoadProject(p)}
                      className="flex-1 text-left px-2 py-2 min-w-0"
                    >
                      <div className="text-neutral-200 text-sm truncate">{p.name}</div>
                      <div className="text-neutral-500 text-xs mt-0.5">
                        {formatTimestamp(p.created_at)}
                        <span className="text-neutral-600"> ({timeAgo(p.created_at)})</span>
                      </div>
                      <div className="text-neutral-600 text-xs mt-0.5">
                        {p.room_count} room{p.room_count !== 1 ? 's' : ''}
                        {p.scale_px_per_meter != null && ` · ${p.scale_px_per_meter.toFixed(1)} px/m`}
                      </div>
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); setDeleteConfirm(p.id); }}
                      className="px-2 py-2 mr-1 text-neutral-600 hover:text-red-400 transition-colors shrink-0"
                      title="Delete project"
                    >
                      &times;
                    </button>
                  </div>
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
        <div className="text-center space-y-4 w-80">
          <div className="animate-spin w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full mx-auto" />
          <p className="text-neutral-400 text-sm">{progress}</p>
          <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
            <div
              className="bg-sky-500 h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="text-neutral-600 text-xs">{percent}%</p>
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
      <div className="relative flex-1 flex flex-col min-w-0">
        <FloorplanCanvas
          rooms={rooms}
          imageUrl={imageUrl}
          selectedRoomId={selectedRoomId}
          canvasMode={canvasMode}
          onCanvasModeChange={setCanvasMode}
          onRoomSelect={setSelectedRoomId}
          onRoomUpdate={handleRoomPolygonUpdate}
          onRoomDrawn={handleRoomDrawn}
        />
        {/* New room dialog */}
        {newRoomPolygon && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-20">
            <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-4 w-72 space-y-3">
              <h3 className="text-sm font-bold text-neutral-200">New Room</h3>
              <div>
                <label className="block text-neutral-500 text-xs mb-1">Name</label>
                <input
                  value={newRoomName}
                  onChange={(e) => setNewRoomName(e.target.value)}
                  placeholder="Room name"
                  autoFocus
                  className="w-full bg-neutral-800 border border-neutral-600 rounded px-2 py-1.5 text-neutral-200 text-sm"
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCreateRoom(); }}
                />
              </div>
              <div>
                <label className="block text-neutral-500 text-xs mb-1">Type</label>
                <select
                  value={newRoomType}
                  onChange={(e) => setNewRoomType(e.target.value)}
                  className="w-full bg-neutral-800 border border-neutral-600 rounded px-2 py-1.5 text-neutral-200 text-sm"
                >
                  {['office', 'bathroom', 'corridor', 'meeting_room', 'kitchen',
                    'storage', 'lobby', 'elevator', 'stairwell', 'utility', 'other', 'unknown',
                  ].map((t) => (
                    <option key={t} value={t}>{t.replace('_', ' ')}</option>
                  ))}
                </select>
              </div>
              <div className="text-neutral-500 text-xs">
                {newRoomPolygon.length} vertices drawn
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleCreateRoom}
                  className="flex-1 px-3 py-1.5 bg-emerald-600 text-white text-xs rounded hover:bg-emerald-500"
                >
                  Create Room
                </button>
                <button
                  onClick={handleCancelCreateRoom}
                  className="flex-1 px-3 py-1.5 bg-neutral-700 text-neutral-300 text-xs rounded hover:bg-neutral-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      <RoomDetail
        room={selectedRoom}
        scale={scale}
        onUpdate={handleRoomUpdate}
        onDelete={handleRoomDelete}
      />
    </div>
  );
}
