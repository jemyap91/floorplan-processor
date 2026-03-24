import { useState } from 'react';
import { exportData, type Room } from '../api';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

const MODE_LABELS: Record<string, { label: string; color: string }> = {
  gemini: { label: 'Gemini AI', color: 'bg-violet-600/30 text-violet-400' },
  hybrid: { label: 'CV + AI', color: 'bg-sky-600/30 text-sky-400' },
  linedraw: { label: 'Line Drawing', color: 'bg-amber-600/30 text-amber-400' },
  furnished: { label: 'Furnished', color: 'bg-emerald-600/30 text-emerald-400' },
};

interface RoomSidebarProps {
  rooms: Room[];
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string) => void;
  scale: { px_per_meter: number | null; source: string } | null;
  projectId: string | null;
  processMode: string | null;
  onBack?: () => void;
}

export function RoomSidebar({ rooms, selectedRoomId, onRoomSelect, scale, projectId, processMode, onBack }: RoomSidebarProps) {
  const totalArea = rooms.reduce((sum, r) => sum + (r.area_sqm || 0), 0);
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    if (!projectId) return;
    setExporting(true);
    try {
      await exportData(projectId, 'xlsx');
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="w-64 bg-neutral-950 border-r border-neutral-800 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-neutral-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {onBack && (
            <button
              onClick={onBack}
              className="text-neutral-500 hover:text-neutral-300 transition-colors"
              title="Back to projects"
            >
              &larr;
            </button>
          )}
          <h2 className="text-xs font-bold text-sky-400 uppercase tracking-wider">
            Rooms ({rooms.length})
          </h2>
        </div>
        {projectId && (
          <button
            onClick={handleExport}
            disabled={exporting || rooms.length === 0}
            className="px-2 py-1 text-xs bg-emerald-700 hover:bg-emerald-600 disabled:bg-neutral-700 disabled:text-neutral-500 text-white rounded transition-colors"
            title="Export to Excel"
          >
            {exporting ? 'Exporting...' : 'Export'}
          </button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {rooms.map((room, i) => (
          <button
            key={room.id}
            onClick={() => onRoomSelect(room.id)}
            className={`w-full text-left p-2 rounded-md transition-colors ${
              room.id === selectedRoomId ? 'bg-neutral-800' : 'hover:bg-neutral-900'
            }`}
            style={{ borderLeft: `3px solid ${ROOM_COLORS[i % ROOM_COLORS.length]}` }}
          >
            <div className="text-sm text-neutral-200">{room.name || 'Unnamed'}</div>
            {room.unit_name && (
              <div className="text-xs text-neutral-600">{room.unit_name}</div>
            )}
            <div className="text-xs text-neutral-500">
              {room.area_sqm ? `${room.area_sqm.toFixed(1)} m²` : '—'}
              {room.perimeter_m ? ` · ${room.perimeter_m.toFixed(1)}m` : ''}
            </div>
            {room.name === 'Unnamed' && (
              <div className="text-xs text-amber-500 mt-0.5">No label detected</div>
            )}
          </button>
        ))}
      </div>
      <div className="p-3 border-t border-neutral-800 text-xs text-neutral-500 space-y-1">
        {processMode && MODE_LABELS[processMode] && (
          <div className="flex items-center gap-1.5">
            <span>Mode:</span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${MODE_LABELS[processMode].color}`}>
              {MODE_LABELS[processMode].label}
            </span>
          </div>
        )}
        <div>
          Scale: {scale?.px_per_meter
            ? `${(1 / scale.px_per_meter).toFixed(4)}m/px (${scale.source})`
            : 'Not set'}
        </div>
        <div>Total area: {totalArea.toFixed(1)} m²</div>
      </div>
    </div>
  );
}
