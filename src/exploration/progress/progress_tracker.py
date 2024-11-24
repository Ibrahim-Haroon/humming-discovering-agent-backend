from typing import Dict
from threading import RLock
from datetime import datetime
from src.exploration.progress.exploration_stats import ExplorationStats


class ProgressTracker:
    """Tracks progress of conversation exploration"""

    def __init__(self):
        self.__lock = RLock()
        self.__stats = ExplorationStats()
        self.__active_workers = 0
        self.__paths_in_progress: Dict[str, int] = {}  # node_id -> depth

    def worker_started(self) -> None:
        """Record that a worker has started"""
        with self.__lock:
            self.__active_workers += 1

    def worker_finished(self) -> None:
        """Record that a worker has finished"""
        with self.__lock:
            self.__active_workers -= 1

    def path_started(self, node_id: str, depth: int) -> None:
        """Record start of path exploration"""
        with self.__lock:
            self.__paths_in_progress[node_id] = depth
            self.__stats.max_depth_reached = max(
                self.__stats.max_depth_reached,
                depth
            )

    def path_finished(self, node_id: str, is_terminal: bool) -> None:
        """Record completion of path exploration"""
        with self.__lock:
            if node_id in self.__paths_in_progress:
                del self.__paths_in_progress[node_id]
            self.__stats.total_nodes += 1
            if is_terminal:
                self.__stats.terminal_nodes += 1

    def edge_added(self) -> None:
        """Record addition of new edge"""
        with self.__lock:
            self.__stats.total_edges += 1

    def is_complete(self) -> bool:
        """Check if exploration is complete"""
        with self.__lock:
            return (
                    self.__active_workers == 0
                    and len(self.__paths_in_progress) == 0
            )

    def mark_complete(self) -> None:
        """Mark exploration as complete"""
        with self.__lock:
            self.__stats.end_time = datetime.now()

    @property
    def stats(self) -> ExplorationStats:
        """Get current exploration stats"""
        with self.__lock:
            return ExplorationStats(**self.__stats.__dict__)

    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        stats = self.stats
        return f"""
        Exploration Progress:
        - Nodes explored: {stats.total_nodes}
        - Terminal paths found: {stats.terminal_nodes}
        - Total transitions: {stats.total_edges}
        - Max depth: {stats.max_depth_reached}
        - Active workers: {self.__active_workers}
        - Paths in progress: {len(self.__paths_in_progress)}
        - Duration: {stats.duration:.1f}s
        """
