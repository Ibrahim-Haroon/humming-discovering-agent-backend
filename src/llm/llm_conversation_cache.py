from threading import RLock
from collections.abc import MutableMapping
from typing import Dict, List, Tuple, Optional, Iterator, override
from src.util.singleton import singleton


@singleton
class LlmConversationCache(MutableMapping):
    def __init__(self):
        """
        A thread-safe cache for storing conversations. Key is an integer representing the LLM ID.
        The value is a list of tuples, where the first element is the message from the voice AI agent
        and the second element is the LLM response.
        """
        self.__cache: Dict[int, List[Tuple[str, str]]] = {}
        self._lock = RLock()

    @override
    def __getitem__(self, llm_id: int) -> Optional[List[Tuple[str, str]]]:
        with self._lock:
            return self.__cache.get(llm_id)

    @override
    def __setitem__(self, llm_id: int, value: Tuple[str, str]) -> None:
        with self._lock:
            if llm_id not in self.__cache:
                self.__cache[llm_id] = []
            self.__cache[llm_id].append(value)

    @override
    def __delitem__(self, llm_id: int) -> None:
        with self._lock:
            if llm_id in self.__cache:
                del self.__cache[llm_id]

    @override
    def __iter__(self) -> Iterator[int]:
        with self._lock:
            return iter(self.__cache.keys())

    @override
    def __len__(self) -> int:
        with self._lock:
            return len(self.__cache)

    def items(self) -> List[Tuple[int, List[Tuple[str, str]]]]:
        """Return all items in the cache as a list of (key, value) pairs."""
        with self._lock:
            return list(self.__cache.items())

    def clear(self) -> None:
        """Clears the entire cache."""
        with self._lock:
            self.__cache.clear()

    def keys(self) -> List[int]:
        """Returns a list of all LLM IDs in the cache."""
        with self._lock:
            return list(self.__cache.keys())

    def values(self) -> List[List[Tuple[str, str]]]:
        """Returns a list of all values (conversations) in the cache."""
        with self._lock:
            return list(self.__cache.values())
