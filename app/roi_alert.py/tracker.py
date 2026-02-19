from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import BBox
from .vision import iou


@dataclass
class Track:
    tid: int
    bbox: BBox
    last_seen: float
    in_roi_since: Optional[float] = None
    fired: bool = False


class IoUTracker:
    def __init__(self, match_thres: float, ttl_sec: float):
        self.match_thres = match_thres
        self.ttl_sec = ttl_sec
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[BBox], ts: float) -> List[Track]:
        dead = [tid for tid, tr in self.tracks.items() if (ts - tr.last_seen) > self.ttl_sec]
        for tid in dead:
            del self.tracks[tid]

        assigned = set()

        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = None
            for j, bb in enumerate(dets):
                if j in assigned:
                    continue
                val = iou(tr.bbox, bb)
                if val > best_iou:
                    best_iou = val
                    best_j = j
            if best_j is not None and best_iou >= self.match_thres:
                tr.bbox = dets[best_j]
                tr.last_seen = ts
                assigned.add(best_j)

        for j, bb in enumerate(dets):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid=tid, bbox=bb, last_seen=ts)

        return list(self.tracks.values())