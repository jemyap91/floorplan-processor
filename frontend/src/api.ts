import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface Room {
  id: string;
  name: string;
  room_type: string;
  boundary_polygon: number[][];
  area_sqm: number | null;
  perimeter_m: number | null;
  boundary_lengths_m: number[] | null;
  centroid: [number, number];
  source: string;
  confidence: number;
}

export interface ProcessResult {
  project_id: string;
  rooms: Room[];
  excluded_regions: { bbox: number[] }[];
  scale: { px_per_meter: number | null; source: string };
  image_size: { width: number; height: number };
}

export async function processFloorplan(file: File, pageNum = 0): Promise<ProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post(`/process?page_num=${pageNum}`, formData);
  return data;
}

export async function getProjects() {
  const { data } = await api.get('/projects');
  return data;
}

export async function getRooms(projectId: string): Promise<Room[]> {
  const { data } = await api.get(`/projects/${projectId}/rooms`);
  return data;
}

export async function updateRoom(roomId: string, update: Partial<Room>) {
  const { data } = await api.put(`/rooms/${roomId}`, update);
  return data;
}

export async function deleteRoom(roomId: string) {
  const { data } = await api.delete(`/rooms/${roomId}`);
  return data;
}

export async function updateScale(projectId: string, pxPerMeter: number) {
  const { data } = await api.put(`/projects/${projectId}/scale?px_per_meter=${pxPerMeter}`);
  return data;
}

export async function exportData(projectId: string, format: 'json' | 'csv') {
  const { data } = await api.get(`/export/${projectId}?format=${format}`);
  return data;
}

export function getImageUrl(projectId: string) {
  return `/api/projects/${projectId}/image`;
}
