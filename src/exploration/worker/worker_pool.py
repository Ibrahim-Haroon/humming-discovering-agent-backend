from concurrent.futures import ThreadPoolExecutor, Future, wait
from queue import Queue, Empty
from threading import RLock, Event, Semaphore
from typing import Optional, Dict
from weakref import WeakSet
from datetime import datetime, timedelta
from .conversation_worker import ConversationWorker
from .worker_context import WorkerContext
from ..exploration_task import ExplorationTask


class WorkerPool:
    """Thread-safe pool of conversation workers for parallel exploration"""

    def __init__(
            self,
            workers: list[ConversationWorker],
            max_workers: int = 3,
            max_depth: int = 10,
            task_timeout: int = 300  # 5 minute timeout
    ):
        """
        Initialize the worker pool

        :param workers: List of conversation workers
        :param max_workers: Maximum number of concurrent workers
        :param max_depth: Maximum exploration depth
        :param task_timeout: Maximum time in seconds for a task to complete
        """
        self.__workers = workers
        self.__max_depth = max_depth
        self.__task_timeout = task_timeout

        # Thread synchronization
        self.__task_lock = RLock()
        self.__worker_semaphore = Semaphore(max_workers)
        self.__shutdown_event = Event()

        # Task management
        self.__task_queue: Queue[ExplorationTask] = Queue()
        self.__active_tasks: WeakSet[Future] = WeakSet()
        self.__worker_tasks: Dict[str, datetime] = {}  # worker_id -> start_time

        # Thread pool
        self.__executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="explorer"
        )

    def submit_task(self, task: ExplorationTask) -> None:
        """
        Submit a new exploration task to the pool

        :param task: Task to submit
        """
        if self.__shutdown_event.is_set():
            raise RuntimeError("Worker pool is shutting down")

        with self.__task_lock:
            self.__task_queue.put(task)
            self.__start_next_worker()

    def __start_next_worker(self) -> None:
        """
        Starts the next available worker on a queued task.
        Thread-safe method for worker management.
        """
        try:
            # Try to acquire worker semaphore without blocking
            if not self.__worker_semaphore.acquire(blocking=False):
                return  # No workers available

            with self.__task_lock:
                if self.__task_queue.empty():
                    self.__worker_semaphore.release()
                    return

                task = self.__task_queue.get_nowait()
                worker = self.__get_available_worker()

                if not worker:
                    self.__task_queue.put(task)  # Put task back
                    self.__worker_semaphore.release()
                    return

                # Submit task to thread pool
                future = self.__executor.submit(
                    self.__process_task_with_timeout,
                    worker,
                    task
                )

                # Track active task
                self.__active_tasks.add(future)
                self.__worker_tasks[str(id(worker))] = datetime.now()
                # Add completion callback
                future.add_done_callback(self.__task_completed)
        except Empty:
            self.__worker_semaphore.release()
        except Exception as e:
            self.__worker_semaphore.release()
            print(f"Error starting worker: {str(e)}")

    def __process_task_with_timeout(
            self,
            worker: ConversationWorker,
            task: ExplorationTask
    ) -> None:
        """
        Processes a task with timeout protection

        :param worker: Worker to use
        :param task: Task to process
        """
        try:
            # Set timeout for this task
            start_time = datetime.now()

            while (datetime.now() - start_time) < timedelta(seconds=self.__task_timeout):
                if self.__shutdown_event.is_set():
                    break

                try:
                    self.__process_task(worker, task)
                    break
                except Exception as e:
                    print(f"Task processing error: {str(e)}")
                    # Could add retry logic here if needed
                    break

        finally:
            worker.cleanup()

    def __process_task(
            self,
            worker: ConversationWorker,
            task: ExplorationTask
    ) -> None:
        """
        Process a single exploration task

        :param worker: Worker to use
        :param task: Task to process
        """
        new_node, edge = worker.explore_path(task.context)
        task.node.add_response(edge.response)

        # Queue child explorations if appropriate
        if (not new_node.is_terminal() and
            task.depth < self.__max_depth and
            not self.__shutdown_event.is_set()
        ):
            new_context = WorkerContext(
                phone_number=task.context.phone_number,
                business_type=task.context.business_type,
                current_node=new_node,
                parent_edge=edge
            )

            new_task = ExplorationTask(
                node=new_node,
                depth=task.depth + 1,
                context=new_context
            )

            self.submit_task(new_task)

    def __task_completed(self, future: Future) -> None:
        """
        Callback for task completion

        :param future: Completed future
        """
        with self.__task_lock:
            self.__active_tasks.discard(future)
            self.__worker_semaphore.release()
            self.__start_next_worker()

    def __get_available_worker(self) -> Optional[ConversationWorker]:
        """
        Gets next available worker using timeout checking

        :return: Available worker or None
        """
        current_time = datetime.now()

        for worker in self.__workers:
            worker_id = id(worker)

            # Check if worker is in use and possibly stuck
            last_start = self.__worker_tasks.get(str(worker_id))
            if last_start:
                if (current_time - last_start) > timedelta(seconds=self.__task_timeout):
                    # Worker might be stuck, clean it up
                    worker.cleanup()
                    del self.__worker_tasks[str(worker_id)]
                else:
                    continue

            return worker

        return None

    def shutdown(
            self,
            wait_for_completion: bool = True,
            timeout: Optional[int] = None
    ) -> None:
        """
        Shuts down the worker pool

        :param wait_for_completion: Whether to wait for completion
        :param timeout: Optional timeout for shutdown
        """
        self.__shutdown_event.set()

        if wait_for_completion and self.__active_tasks:
            # Wait for active tasks with timeout
            done, not_done = wait(
                self.__active_tasks,
                timeout=timeout
            )

            # Cancel any remaining tasks
            for future in not_done:
                future.cancel()

        # Cleanup
        try:
            for worker in self.__workers:
                worker.cleanup()
        finally:
            self.__executor.shutdown(wait=wait_for_completion)

    def get_active_task_count(self) -> int:
        """
        Get count of currently active tasks

        :return: Number of active tasks
        """
        return len(self.__active_tasks)

    def is_idle(self) -> bool:
        """
        Check if pool has no active tasks

        :return: True if no active tasks
        """
        return len(self.__active_tasks) == 0
