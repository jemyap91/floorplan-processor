import { useState } from 'react';
import { exportData, type Room } from '../api';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

interface RoomSidebarProps {
  rooms: Room[];
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string) => void;
  scale: { px_per_meter: number | null; source: string } | null;
  projectId: string | null;
}

export function RoomSidebar({ rooms, selectedRoomId, onRoomSelect, scale, projectId }: RoomSidebarProps) {
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
        <h2 className="text-xs font-bold text-sky-400 uppercase tracking-wider">
          Rooms ({rooms.length})
        </h2>
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
