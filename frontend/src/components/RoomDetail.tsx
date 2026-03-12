import { useState, useEffect } from 'react';
import type { Room, ScaleInfo } from '../api';

interface RoomDetailProps {
  room: Room | null;
  scale: ScaleInfo | null;
  onUpdate: (roomId: string, update: Partial<Room>) => void;
  onDelete: (roomId: string) => void;
}

/** Only show wall segments longer than this fraction of perimeter */
const MIN_WALL_FRACTION = 0.03;

export function RoomDetail({ room, scale, onUpdate, onDelete }: RoomDetailProps) {
  const [name, setName] = useState('');
  const [roomType, setRoomType] = useState('');

  useEffect(() => {
    if (room) {
      setName(room.name);
      setRoomType(room.room_type);
    }
  }, [room?.id]);

  if (!room) {
    return (
      <div className="w-56 bg-neutral-950 border-l border-neutral-800 flex items-center justify-center">
        <p className="text-neutral-600 text-sm">Select a room</p>
      </div>
    );
  }

  const handleNameBlur = () => {
    if (name !== room.name) onUpdate(room.id, { name });
  };

  const handleTypeBlur = () => {
    if (roomType !== room.room_type) onUpdate(room.id, { room_type: roomType });
  };

  const pxPerM = scale?.px_per_meter;

  // Filter wall segments to only show meaningful ones (not tiny noise segments)
  const wallSegments: { index: number; lengthM: number | null; lengthPx: number }[] = [];
  if (room.boundary_lengths_px) {
    const perimeterPx = room.perimeter_px || room.boundary_lengths_px.reduce((a, b) => a + b, 0);
    const minLen = perimeterPx * MIN_WALL_FRACTION;
    room.boundary_lengths_px.forEach((px, i) => {
      if (px >= minLen) {
        wallSegments.push({
          index: i,
          lengthPx: px,
          lengthM: room.boundary_lengths_m?.[i] ?? null,
        });
      }
    });
  }

  // Compute labelled area variance if we have both values
  // (area_sqm from CV scale vs. a future labelled area)
  const totalWallSegments = room.boundary_lengths_px?.length ?? 0;
  const shownWallSegments = wallSegments.length;

  return (
    <div className="w-56 bg-neutral-950 border-l border-neutral-800 p-3 overflow-y-auto">
      <h2 className="text-xs font-bold text-sky-400 uppercase tracking-wider mb-4">
        Room Details
      </h2>
      <div className="space-y-4 text-sm">
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={handleNameBlur}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 text-sm"
          />
        </div>
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Type</label>
          <select
            value={roomType}
            onChange={(e) => { setRoomType(e.target.value); }}
            onBlur={handleTypeBlur}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 text-sm"
          >
            {['office', 'bathroom', 'corridor', 'meeting_room', 'kitchen',
              'storage', 'lobby', 'elevator', 'stairwell', 'utility', 'other', 'unknown',
            ].map((t) => (
              <option key={t} value={t}>{t.replace('_', ' ')}</option>
            ))}
          </select>
        </div>

        {/* Area */}
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Area</label>
          <div className="text-neutral-200">
            {room.area_sqm ? `${room.area_sqm.toFixed(2)} m²` : '—'}
          </div>
          <div className="text-neutral-500 text-xs mt-0.5">
            {room.area_px.toLocaleString(undefined, { maximumFractionDigits: 0 })} px²
          </div>
        </div>

        {/* Perimeter */}
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Perimeter</label>
          <div className="text-neutral-200">
            {room.perimeter_m ? `${room.perimeter_m.toFixed(2)} m` : '—'}
          </div>
          <div className="text-neutral-500 text-xs mt-0.5">
            {room.perimeter_px.toLocaleString(undefined, { maximumFractionDigits: 0 })} px
          </div>
        </div>

        {/* Scale info */}
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Scale</label>
          <div className="text-neutral-400 text-xs">
            {pxPerM
              ? `${pxPerM.toFixed(1)} px/m (${scale?.source})`
              : 'Not detected'}
          </div>
          {pxPerM && (
            <div className="text-neutral-600 text-xs mt-0.5">
              1 px = {(1 / pxPerM * 1000).toFixed(1)} mm | 1 m = {pxPerM.toFixed(0)} px
            </div>
          )}
        </div>

        {/* Wall segments — filtered to real walls only */}
        {wallSegments.length > 0 && (
          <div>
            <label className="block text-neutral-500 text-xs mb-1">
              Wall Segments ({shownWallSegments}{totalWallSegments > shownWallSegments ? ` of ${totalWallSegments}` : ''})
            </label>
            <div className="text-neutral-300 text-xs space-y-0.5">
              {wallSegments.map((seg, i) => (
                <div key={i} className="flex justify-between">
                  <span>Wall {i + 1}:</span>
                  <span>
                    {seg.lengthM != null ? `${seg.lengthM.toFixed(2)}m` : `${seg.lengthPx.toFixed(0)}px`}
                  </span>
                </div>
              ))}
            </div>
            {totalWallSegments > shownWallSegments && (
              <div className="text-neutral-600 text-xs mt-1">
                {totalWallSegments - shownWallSegments} tiny segments hidden (&lt;3% of perimeter)
              </div>
            )}
          </div>
        )}

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Vertices</label>
          <div className="text-neutral-400 text-xs">
            {room.boundary_polygon?.length || 0} points
          </div>
        </div>
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Source</label>
          <div className={`text-xs ${room.source === 'cv' ? 'text-green-400' : 'text-blue-400'}`}>
            {room.source === 'cv' ? 'Auto-detected (CV)' : room.source}
          </div>
        </div>
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Confidence</label>
          <div className="text-neutral-400 text-xs">
            {(room.confidence * 100).toFixed(0)}%
          </div>
        </div>
        <button
          onClick={() => onDelete(room.id)}
          className="w-full mt-4 px-3 py-1.5 bg-red-900/30 text-red-400 text-xs rounded hover:bg-red-900/50"
        >
          Delete Room
        </button>
      </div>
    </div>
  );
}
