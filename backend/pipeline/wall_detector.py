"""Detect wall segments from binary floorplan images."""
import cv2
import numpy as np

def detect_walls(binary: np.ndarray, min_wall_length: int = 30, wall_thickness_range: tuple = (3, 30)) -> dict:
    h, w = binary.shape
    wall_mask = np.zeros_like(binary)
    h_kernel_len = max(min_wall_length, w // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel_len = max(min_wall_length, h // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    wall_mask = cv2.bitwise_or(h_lines, v_lines)
    segments = []
    lines = cv2.HoughLinesP(wall_mask, rho=1, theta=np.pi/180, threshold=min_wall_length, minLineLength=min_wall_length, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 20 or angle > 160:
                orientation = "horizontal"
            elif 70 < angle < 110:
                orientation = "vertical"
            else:
                orientation = "diagonal"
            segments.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "orientation": orientation, "length": float(length)})
    segments = _merge_segments(segments, merge_threshold=15)
    return {"segments": segments, "wall_mask": wall_mask}

def _merge_segments(segments: list, merge_threshold: int = 15) -> list:
    if not segments:
        return segments
    merged = []
    used = set()
    for i, s1 in enumerate(segments):
        if i in used:
            continue
        group = [s1]
        used.add(i)
        for j, s2 in enumerate(segments):
            if j in used or j <= i:
                continue
            if s1["orientation"] != s2["orientation"]:
                continue
            if s1["orientation"] == "horizontal":
                y_dist = abs((s1["y1"]+s1["y2"])/2 - (s2["y1"]+s2["y2"])/2)
                if y_dist < merge_threshold:
                    group.append(s2)
                    used.add(j)
            elif s1["orientation"] == "vertical":
                x_dist = abs((s1["x1"]+s1["x2"])/2 - (s2["x1"]+s2["x2"])/2)
                if x_dist < merge_threshold:
                    group.append(s2)
                    used.add(j)
        all_x = [s["x1"] for s in group] + [s["x2"] for s in group]
        all_y = [s["y1"] for s in group] + [s["y2"] for s in group]
        if group[0]["orientation"] == "horizontal":
            merged_seg = {"x1": min(all_x), "y1": int(np.mean([s["y1"] for s in group])), "x2": max(all_x), "y2": int(np.mean([s["y2"] for s in group])), "orientation": "horizontal"}
        elif group[0]["orientation"] == "vertical":
            merged_seg = {"x1": int(np.mean([s["x1"] for s in group])), "y1": min(all_y), "x2": int(np.mean([s["x2"] for s in group])), "y2": max(all_y), "orientation": "vertical"}
        else:
            merged_seg = group[0]
        merged_seg["length"] = float(np.sqrt((merged_seg["x2"]-merged_seg["x1"])**2 + (merged_seg["y2"]-merged_seg["y1"])**2))
        merged.append(merged_seg)
    return merged
