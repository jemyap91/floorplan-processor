import { useState, useEffect } from 'react';
import type { Room } from '../api';

interface RoomDetailProps {
  room: Room | null;
  onUpdate: (roomId: string, update: Partial<Room>) => void;
  onDelete: (roomId: string) => void;
}

export function RoomDetail({ room, onUpdate, onDelete }: RoomDetailProps) {
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
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Area</label>
          <div className="text-neutral-200">
            {room.area_sqm ? `${room.area_sqm.toFixed(2)} m²` : '—'}
          </div>
        </div>
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Perimeter</label>
          <div className="text-neutral-200">
            {room.perimeter_m ? `${room.perimeter_m.toFixed(2)} m` : '—'}
          </div>
        </div>
        {room.boundary_lengths_m && (
          <div>
            <label className="block text-neutral-500 text-xs mb-1">Wall Segments</label>
            <div className="text-neutral-300 text-xs space-y-0.5">
              {room.boundary_lengths_m.map((len, i) => (
                <div key={i}>Wall {i + 1}: {len.toFixed(2)}m</div>
              ))}
            </div>
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
