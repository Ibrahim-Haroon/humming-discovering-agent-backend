from queue import Queue
from typing import Dict
from threading import RLock
from src.util.singleton import singleton


@singleton
class WebhookCallback:
    def __init__(self):
        self.callbacks: Dict[str, Queue] = {}
        self.callback_lock = RLock()
        self.ngrok_tunnel = None
