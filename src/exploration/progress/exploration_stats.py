from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ExplorationStats:
    """Statistics about the exploration progress"""
    total_nodes: int = 0
    terminal_nodes: int = 0
    total_edges: int = 0
    max_depth_reached: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()