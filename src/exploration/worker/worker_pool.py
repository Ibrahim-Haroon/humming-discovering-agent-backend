import logging
from queue import Queue, Empty
from threading import RLock, Event, Semaphore
from typing import Optional, Dict, Set
from weakref import WeakSet
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future, wait
from .conversation_worker import ConversationWorker
from .worker_context import WorkerContext
from ..exploration_task import ExplorationTask
from ...core.model.conversation_graph import ConversationGraph


class WorkerPool:
    """Thread-safe pool of conversation workers for parallel exploration"""
    logger = logging.getLogger(__name__)

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

        # Track tasks by business context
        self.__business_contexts: Dict[str, Set[str]] = {}  # business_type -> set of worker_ids
        self.__worker_tasks: Dict[str, datetime] = {}  # worker_id -> start_time

        # Thread pool
        self.__executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="explorer"
        )

    @property
    def task_queue(self):
        return self.__task_queue

    def submit_task(self, task: ExplorationTask) -> None:
        """
        Submit a new exploration task to the pool

        :param task: Task to submit
        """
        if self.__shutdown_event.is_set():
            raise RuntimeError("Worker pool is shutting down")

        with self.__task_lock:
            self.__task_queue.put(task)
            # Initialize business context tracking if needed
            if task.context.business_type not in self.__business_contexts:
                self.__business_contexts[task.context.business_type] = set()
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
                worker = self.__get_available_worker(task.context.business_type)

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
                worker_id = str(id(worker))
                self.__active_tasks.add(future)
                self.__worker_tasks[worker_id] = datetime.now()
                self.__business_contexts[task.context.business_type].add(worker_id)

                # Add completion callback
                future.add_done_callback(lambda f: self.__task_completed(f, task.context.business_type, worker_id))

        except Empty:
            self.__worker_semaphore.release()
        except Exception as e:
            self.__worker_semaphore.release()
            WorkerPool.logger.error(f"Error starting worker: {str(e)}")

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
                    WorkerPool.logger.error(f"Task processing error: {str(e)}")
                    break

        finally:
            worker.cleanup(task.context)  # Pass context for cleanup

    def __process_task(self, worker: ConversationWorker, task: ExplorationTask) -> None:
        new_node, edge = worker.explore_path(task.context)
        task.node.add_response(edge.response)

        print(f"\nNode {new_node.id} reached state: {new_node.state}")

        if new_node.is_terminal():
            print(f"Terminal state reached, attempting backtrack from node {task.context.current_node.id}")
            parent_node = task.context.current_node
            while parent_node and not parent_node.is_terminal():
                print(f"Checking parent node {parent_node.id} for unexplored paths")
                prompt = worker.generate_new_prompt(parent_node)
                print(f"Generated prompt: {prompt}")

                if prompt and not parent_node.has_similar_response(prompt):
                    print(f"Found new path to explore from node {parent_node.id}")
                    new_context = WorkerContext(
                        phone_number=task.context.phone_number,
                        business_type=task.context.business_type,
                        current_node=parent_node,
                        metadata={'prompt': prompt}  # Store prompt for worker
                    )
                    new_task = ExplorationTask(
                        node=parent_node,
                        depth=task.depth,
                        context=new_context
                    )
                    self.submit_task(new_task)
                    break
                else:
                    print(f"No new paths found from node {parent_node.id}")
                parent_node = ConversationGraph().get_node(parent_node.parent_id)

            if parent_node is None:
                print("Exploration complete - all paths from root explored")

        elif task.depth < self.__max_depth and not self.__shutdown_event.is_set():
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
        else:
            print(f"Max depth {self.__max_depth} reached")

    def __task_completed(
            self,
            future: Future,
            business_type: str,
            worker_id: str
    ) -> None:
        """
        Callback for task completion

        :param future: Completed future
        :param business_type: Business type of completed task
        :param worker_id: ID of worker that completed the task
        """
        with self.__task_lock:
            self.__active_tasks.discard(future)
            if worker_id in self.__worker_tasks:
                del self.__worker_tasks[worker_id]
            if business_type in self.__business_contexts:
                self.__business_contexts[business_type].discard(worker_id)
            self.__worker_semaphore.release()
            self.__start_next_worker()

    def __get_available_worker(self, business_type: str) -> Optional[ConversationWorker]:
        """
        Gets next available worker using timeout checking and business context

        :param business_type: Business type to get worker for
        :return: Available worker or None
        """
        current_time = datetime.now()

        for worker in self.__workers:
            worker_id = str(id(worker))

            # Check if worker is in use and possibly stuck
            last_start = self.__worker_tasks.get(worker_id)
            if last_start:
                if (current_time - last_start) > timedelta(seconds=self.__task_timeout):
                    # Worker might be stuck, clean it up
                    context = WorkerContext(
                        phone_number="",  # Not needed for cleanup
                        business_type=business_type,
                        current_node=None  # Not needed for cleanup
                    )
                    worker.cleanup(context)

                    # Clean up tracking
                    del self.__worker_tasks[worker_id]
                    for contexts in self.__business_contexts.values():
                        contexts.discard(worker_id)
                else:
                    continue

            # Check if worker is already handling this business type
            if worker_id in self.__business_contexts.get(business_type, set()):
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
        if not self.__task_queue.empty() or self.get_active_task_count() > 0:
            return

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

        # Cleanup all workers with their respective contexts
        try:
            for worker in self.__workers:
                for business_type in self.__business_contexts.keys():
                    context = WorkerContext(
                        phone_number="",  # Not needed for cleanup
                        business_type=business_type,
                        current_node=None  # Not needed for cleanup
                    )
                    worker.cleanup(context)
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
        return len(self.__active_tasks) == 0 and self.__task_queue.empty()
