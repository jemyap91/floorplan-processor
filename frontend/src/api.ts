import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface Room {
  id: string;
  name: string;
  room_type: string;
  boundary_polygon: number[][];
  area_px: number;
  perimeter_px: number;
  area_sqm: number | null;
  perimeter_m: number | null;
  boundary_lengths_px: number[];
  boundary_lengths_m: number[] | null;
  centroid: [number, number];
  source: string;
  confidence: number;
  unit_name: string | null;
  printed_area_sqm: number | null;
  area_divergence_flag: boolean;
}

export interface ScaleInfo {
  px_per_meter: number | null;
  source: string;
}

export interface ProcessResult {
  project_id: string;
  rooms: Room[];
  excluded_regions: { bbox: number[] }[];
  scale: ScaleInfo;
  image_size: { width: number; height: number };
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
  scale_px_per_meter: number | null;
  scale_source: string;
  room_count: number;
  process_mode: string;
}

export type ProcessMode = 'hybrid' | 'gemini' | 'linedraw' | 'furnished';

export async function processFloorplan(
  file: File, pageNum = 0, mode: ProcessMode = 'hybrid',
  jobId?: string, filterColors = true, geminiModel = 'flash',
): Promise<ProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', mode);
  formData.append('gemini_model', geminiModel);
  if (jobId) formData.append('job_id', jobId);
  if (mode === 'linedraw') formData.append('filter_colors', filterColors ? 'true' : 'false');
  const { data } = await api.post(`/process?page_num=${pageNum}`, formData);
  return data;
}

export interface ProgressInfo {
  percent: number;
  message: string;
}

export async function getProgress(jobId: string): Promise<ProgressInfo> {
  const { data } = await api.get(`/progress/${jobId}`);
  return data;
}

export async function getProjects(): Promise<Project[]> {
  const { data } = await api.get('/projects');
  return data;
}

export async function getProject(projectId: string): Promise<Project> {
  const { data } = await api.get(`/projects/${projectId}`);
  return data;
}

export async function deleteProject(projectId: string) {
  const { data } = await api.delete(`/projects/${projectId}`);
  return data;
}

export async function getRooms(projectId: string): Promise<Room[]> {
  const { data } = await api.get(`/projects/${projectId}/rooms`);
  return data;
}

export async function createRoom(projectId: string, name: string, roomType: string, polygon: number[][]): Promise<Room> {
  const { data } = await api.post('/rooms', {
    project_id: projectId,
    name,
    room_type: roomType,
    boundary_polygon: polygon,
  });
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

export async function exportData(projectId: string, format: 'json' | 'csv' | 'xlsx') {
  if (format === 'xlsx') {
    const response = await api.get(`/export/${projectId}?format=xlsx`, {
      responseType: 'blob',
    });
    const blob = new Blob([response.data], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `floorplan-export.xlsx`;
    a.click();
    URL.revokeObjectURL(url);
    return;
  }
  const { data } = await api.get(`/export/${projectId}?format=${format}`);
  return data;
}

export function getImageUrl(projectId: string) {
  return `/api/projects/${projectId}/image`;
}
