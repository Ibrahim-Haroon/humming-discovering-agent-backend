import time
from threading import Event
from typing import List
from src.core.model.conversation_node import ConversationNode
from src.core.model.conversation_graph import ConversationGraph
from src.core.enum.conversation_state import ConversationState
from .worker.conversation_worker import ConversationWorker
from .worker.worker_context import WorkerContext
from .worker.worker_pool import WorkerPool, ExplorationTask
from .progress.progress_tracker import ProgressTracker


class ConversationExplorer:
    """
    Orchestrates the parallel exploration of conversation paths.
    """

    def __init__(
            self,
            workers: List[ConversationWorker],
            phone_number: str,
            business_type: str,
            max_depth: int = 10,
            max_parallel: int = 3
    ):
        self.__phone_number = phone_number
        self.__business_type = business_type
        self.__max_depth = max_depth
        self.__stop_event = Event()
        self.__graph = ConversationGraph()
        self.__progress = ProgressTracker()
        self.__worker_pool = WorkerPool(
            workers=workers,
            max_workers=max_parallel,
            max_depth=max_depth
        )

    @property
    def progress(self) -> ProgressTracker:
        """Returns the progress tracker"""
        return self.__progress

    def explore(self) -> ConversationGraph:
        """
        Explores conversation paths in parallel using worker pool.
        Returns the completed conversation graph.
        """
        try:
            initial_node = ConversationNode(
                conversation_transcription="",  # Will be populated by first call
                state=ConversationState.INITIAL
            )
            self.__graph.add_node(initial_node)

            # Create initial context and task
            initial_context = WorkerContext(
                phone_number=self.__phone_number,
                business_type=self.__business_type,
                current_node=initial_node
            )
            initial_task = ExplorationTask(
                node=initial_node,
                depth=0,
                context=initial_context
            )

            # Start exploration
            self.__worker_pool.submit_task(initial_task)

            # Wait for completion or stop event
            while not self.__stop_event.is_set():
                if self.__worker_pool.is_idle() and self.__worker_pool.task_queue.empty():
                    break
                time.sleep(1)

            return self.__graph

        finally:
            self.__worker_pool.shutdown()

    def stop(self) -> None:
        """Signals the explorer to stop after current tasks complete"""
        self.__stop_event.set()