import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import config


@dataclass
class PlayerSeat:
    """Represents a detected player seat"""
    seat_name: str
    seat_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    edge_ratio: float
    laplacian_var: float
    confidence: float  # occupancy confidence score
    is_occupied: bool
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class PlayerDetector:
    """
    Detects players at poker table seats using Canny edge detection + NMS.
    
    Method:
    - Compute edge density (edge_ratio) using Canny edge detection
    - Compute Laplacian variance for texture detail
    - Use dual-threshold: occupied if both metrics exceed thresholds
    - Apply NMS to handle overlapping seat ROIs
    
    This approach is robust to:
    - Single boundary edges (empty seats with felt/background edge)
    - Blurry compression artifacts
    - Overlapping seat ROI definitions
    """
    
    def __init__(self, edge_ratio_threshold: float = 0.1, laplacian_var_threshold: float = 100.0,
                 canny_low: int = 50, canny_high: int = 150, nms_overlap_threshold: float = 0.6):
        """
        Args:
            edge_ratio_threshold: Min edge pixel ratio (0..1) to consider seat occupied.
                                 Typical: 0.05-0.15. Lower = more sensitive.
            laplacian_var_threshold: Min Laplacian variance to consider seat occupied.
                                    Typical: 50-200. Lower = more sensitive.
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            nms_overlap_threshold: NMS suppression threshold (0..1).
                                  Uses intersection / min(areaA, areaB) metric.
                                  Typical: 0.35-0.60. Higher = more aggressive suppression.
        """
        self.edge_ratio_threshold = edge_ratio_threshold
        self.laplacian_var_threshold = laplacian_var_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.nms_overlap_threshold = nms_overlap_threshold
        
        # Load seat coordinates from config
        self.seat_coords = config.SEAT_ROIS

    def set_seat_coords(self, seat_coords: dict):
        """Update seat coordinates. Dict format: {seat_name: (x_pct, y_pct, w_pct, h_pct)}"""
        self.seat_coords = seat_coords

    def _to_grayscale(self, roi: np.ndarray) -> np.ndarray:
        """Convert ROI to grayscale if needed."""
        if len(roi.shape) == 3:
            return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi

    def _calculate_edge_ratio(self, roi_gray: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection.
        
        Returns:
            Ratio of edge pixels to total pixels (0..1)
        """
        if roi_gray.size == 0:
            return 0.0
        
        # Apply Canny edge detection
        edges = cv2.Canny(roi_gray, self.canny_low, self.canny_high)
        
        # Count edge pixels
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        
        return float(edge_pixels) / total_pixels

    def _calculate_laplacian_variance(self, roi_gray: np.ndarray) -> float:
        """
        Calculate Laplacian variance (sharpness/texture metric).
        High values indicate fine detail (text, avatar).
        Low values indicate uniform regions (felt).
        
        Returns:
            Laplacian variance
        """
        if roi_gray.size == 0:
            return 0.0
        
        # Apply Laplacian operator (detects sharp transitions/detail)
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        
        # Return variance of Laplacian response
        return float(np.var(laplacian))

    def _calculate_confidence(self, edge_ratio: float, laplacian_var: float) -> float:
        """
        Calculate occupancy confidence score (0..1) based on both metrics.
        Used for NMS ranking.
        
        Args:
            edge_ratio: Edge density metric
            laplacian_var: Laplacian variance metric
        
        Returns:
            Confidence score (0..1)
        """
        # Normalize both metrics to 0..1 range (simple approach)
        edge_score = min(edge_ratio / 0.3, 1.0)  # normalize assuming max ~0.3
        lap_score = min(laplacian_var / 500.0, 1.0)  # normalize assuming max ~500
        
        # Combine as average
        return (edge_score + lap_score) / 2.0

    def _compute_intersection_over_min(self, box_a: Tuple[int, int, int, int], 
                                       box_b: Tuple[int, int, int, int]) -> float:
        """
        Compute intersection / min(area_a, area_b) metric.
        Better for overlapping ROIs than IoU.
        
        Args:
            box_a: (x1, y1, x2, y2)
            box_b: (x1, y1, x2, y2)
        
        Returns:
            Overlap metric (0..1)
        """
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b
        
        # Intersection area
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Min area
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        min_area = min(area_a, area_b)
        
        if min_area == 0:
            return 0.0
        
        return intersection / min_area

    def _apply_nms(self, candidates: List[PlayerSeat]) -> List[PlayerSeat]:
        """
        Apply non-maximum suppression to overlapping seat detections.
        
        Args:
            candidates: List of PlayerSeat candidates
        
        Returns:
            Filtered list with overlapping detections suppressed
        """
        if len(candidates) <= 1:
            return candidates
        
        # Sort by confidence (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, cand_i in enumerate(sorted_candidates):
            if i in suppressed:
                continue
            
            keep.append(cand_i)
            
            # Check for overlaps with remaining candidates
            box_i = (cand_i.x1, cand_i.y1, cand_i.x2, cand_i.y2)
            for j in range(i + 1, len(sorted_candidates)):
                if j in suppressed:
                    continue
                
                cand_j = sorted_candidates[j]
                box_j = (cand_j.x1, cand_j.y1, cand_j.x2, cand_j.y2)
                
                overlap = self._compute_intersection_over_min(box_i, box_j)
                
                if overlap > self.nms_overlap_threshold:
                    suppressed.add(j)
        
        return keep
    
    def _apply_exclusion_groups(self, occupied: List[PlayerSeat]) -> List[PlayerSeat]:
        """
        Enforce mutual exclusion: in each group, keep at most 1 occupied seat
        (the one with highest confidence).
        """
        if not hasattr(config, "SEAT_EXCLUSION_GROUPS"):
            return occupied

        by_name = {p.seat_name: p for p in occupied}
        to_remove = set()

        for group in config.SEAT_EXCLUSION_GROUPS:
            present = [by_name[name] for name in group if name in by_name]
            if len(present) <= 1:
                continue

        # keep best
            best = max(present, key=lambda p: p.confidence)
            for p in present:
                if p.seat_name != best.seat_name:
                    to_remove.add(p.seat_name)

        return [p for p in occupied if p.seat_name not in to_remove]

    def detect(self, frame: np.ndarray, table_box) -> List[PlayerSeat]:
        """
        Detect players at all seat positions using dual-threshold method + NMS.
        
        Args:
            frame: Input frame (BGR image)
            table_box: TableBox object defining table boundaries
            
        Returns:
            List of PlayerSeat objects with occupancy status (NMS-filtered)
        """
        candidates = []
        
        for seat_idx, (seat_name, (x_pct, y_pct, w_pct, h_pct)) in enumerate(self.seat_coords.items()):
            # Get pixel coordinates from relative coordinates
            x1, y1, x2, y2 = table_box.roi_from_rel(x_pct, y_pct, w_pct, h_pct)
            
            # Clip to frame boundaries
            x1 = max(0, min(x1, frame.shape[1]))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            roi_gray = self._to_grayscale(roi)
            
            # Calculate metrics
            edge_ratio = self._calculate_edge_ratio(roi_gray)
            laplacian_var = self._calculate_laplacian_variance(roi_gray)
            confidence = self._calculate_confidence(edge_ratio, laplacian_var)
            
            # Dual-threshold: occupied if BOTH metrics exceed thresholds
            is_occupied = (edge_ratio > self.edge_ratio_threshold and 
                          laplacian_var > self.laplacian_var_threshold)
            
            player = PlayerSeat(
                seat_name=seat_name,
                seat_id=seat_idx,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                edge_ratio=edge_ratio,
                laplacian_var=laplacian_var,
                confidence=confidence,
                is_occupied=is_occupied
            )
            candidates.append(player)
        
        # Apply NMS to occupied candidates
        occupied_candidates = [c for c in candidates if c.is_occupied]
        occupied_candidates = self._apply_exclusion_groups(occupied_candidates)
        filtered_occupied = self._apply_nms(occupied_candidates)
        
        # Combine filtered occupied with all unoccupied for complete picture
        unoccupied = [c for c in candidates if not c.is_occupied]
        result = filtered_occupied + unoccupied
        
        # Sort by seat_id for consistent ordering
        result.sort(key=lambda x: x.seat_id)
        
        return result

    def detect_occupied_seats(self, frame: np.ndarray, table_box) -> List[Tuple[str, int]]:
        """
        Convenience method to get just the occupied seat info (after NMS).
        
        Returns:
            List of tuples (seat_name, seat_id) that are occupied
        """
        players = self.detect(frame, table_box)
        return [(p.seat_name, p.seat_id) for p in players if p.is_occupied]