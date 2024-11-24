from enum import Enum, auto


class ConversationState(Enum):
    """
    Represents the various states a conversation node can be in.

    States:
    - INITIAL: Starting state of a conversation (e.g., greeting)
    - IN_PROGRESS: Active conversation where more inputs are expected
    - TERMINAL_SUCCESS: Conversation ended successfully (e.g., appointment booked)
    - TERMINAL_TRANSFER: Conversation ended with transfer to human agent
    - TERMINAL_FALLBACK: Conversation ended due to AI unable to handle request
    - ERROR: Conversation ended due to system error or invalid state
    """
    INITIAL = auto()
    IN_PROGRESS = auto()
    TERMINAL_SUCCESS = auto()
    TERMINAL_TRANSFER = auto()
    TERMINAL_FALLBACK = auto()
    ERROR = auto()
