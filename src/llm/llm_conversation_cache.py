from threading import RLock
from collections.abc import MutableMapping
from typing import Dict, List, Tuple, Optional, Iterator, override
from uuid import UUID

from src.util.singleton import singleton


@singleton
class LlmConversationCache(MutableMapping):
    """
    A thread-safe cache for storing conversations. Key is a string representing the business context. The value is a
    list of tuples, where the first element is the prompt made by the LLM, and the second element is the Agent and LLM
    conversation's transcription.
    """

    def __init__(self):
        self.__cache: Dict[str, List[Tuple[str, str]]] = {}
        self._lock = RLock()

    @override
    def __getitem__(self, llm_id: str) -> Optional[List[Tuple[str, str]]]:
        with self._lock:
            return self.__cache.get(llm_id)

    @override
    def __setitem__(self, llm_id: str, value: Tuple[str, str]) -> None:
        with self._lock:
            if llm_id not in self.__cache:
                self.__cache[llm_id] = []
            self.__cache[llm_id].append(value)

    @override
    def __delitem__(self, llm_id: str) -> None:
        with self._lock:
            if llm_id in self.__cache:
                del self.__cache[llm_id]

    @override
    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(self.__cache.keys())

    @override
    def __len__(self) -> int:
        with self._lock:
            return len(self.__cache)

    def items(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Return all items in the cache as a list of (key, value) pairs."""
        with self._lock:
            return list(self.__cache.items())

    def clear(self) -> None:
        """Clears the entire cache."""
        with self._lock:
            self.__cache.clear()

    def keys(self) -> List[str]:
        """Returns a list of all LLM IDs in the cache."""
        with self._lock:
            return list(self.__cache.keys())

    def values(self) -> List[List[Tuple[str, str]]]:
        """Returns a list of all values (conversations) in the cache."""
        with self._lock:
            return list(self.__cache.values())
