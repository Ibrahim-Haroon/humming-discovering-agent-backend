import ngrok
import requests
from typing import Optional
from datetime import datetime
from queue import Queue, Empty

from ngrok import set_auth_token

from src.util.env import Env
from src.rest.webhook.webhook_callback import WebhookCallback
from src.rest.webhook.hamming_webhook_server import start_webhook_server
from src.rest.dto.hamming_call_request_dto import HammingCallRequestDTO
from src.rest.dto.hamming_call_response_dto import HammingCallResponseDTO


class VoiceApiError(Exception):
    """Custom exception for Voice API errors"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class VoiceApiClient:
    """Client for interacting with the Hamming Voice API"""
    
    def __init__(self):
        self.__base_url = "https://app.hamming.ai/api".rstrip('/')
        self.__api_key = Env()["HAMMING_API_KEY"]
        self.__headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json"
        }
        self.__callback = WebhookCallback()

        start_webhook_server()
        ngrok.set_auth_token(Env()["NGROK_AUTH_TOKEN"])
        self.__callback.ngrok_tunnel = ngrok.connect(8080)
        self.__webhook_url = f"{self.__callback.ngrok_tunnel.url()}/webhook"

    def start_call(
        self,
        phone_number: str,
        prompt: str
    ) -> HammingCallResponseDTO:
        """
        Initiates a new phone call

        :param phone_number: Target phone number
        :param prompt: System prompt for the agent
        :return: Call response containing call ID
        :raises VoiceApiError: If API call fails
        """
        try:
            callback_queue = Queue()

            request = HammingCallRequestDTO(
                phone_number=phone_number,
                prompt=prompt,
                webhook_url=self.__webhook_url
            )

            with self.__callback.callback_lock:
                response = requests.post(
                    f"{self.__base_url}/rest/exercise/start-call",
                    headers=self.__headers,
                    json=request.__dict__,
                    timeout=30
                )

                if response.status_code != 200:
                    raise VoiceApiError(
                        f"Failed to start call: {response.text}",
                        response.status_code
                    )

                data = response.json()
                call_id = data["id"]
                self.__callback.callbacks[call_id] = callback_queue

            return HammingCallResponseDTO(id=call_id)

        except requests.RequestException as e:
            raise VoiceApiError(f"API request failed: {str(e)}")

    def get_recording(
        self,
        call_id: str,
        timeout: int = 300
    ) -> str:
        """
        Gets the recording for a completed call

        :param call_id: ID of the call
        :param timeout: Maximum time to wait for recording in seconds
        :return: Path to downloaded recording
        :raises VoiceApiError: If recording retrieval fails
        """
        try:
            # Wait for webhook callback
            with self.__callback.callback_lock:
                if call_id not in self.__callback.callbacks:
                    raise VoiceApiError("No callback queue for call ID")
                queue = self.__callback.callbacks[call_id]

            # Wait for callback
            try:
                webhook_data = queue.get(timeout=timeout)
                if not webhook_data.get('recording_available'):
                    raise VoiceApiError("Recording not available")
            except Empty:
                raise VoiceApiError("Webhook timeout")
            finally:
                with self.__callback.callback_lock:
                    if call_id in self.__callback.callbacks:
                        del self.__callback.callbacks[call_id]

            # Get recording
            response = requests.get(
                f"{self.__base_url}/media/exercise",
                headers=self.__headers,
                params={"id": call_id},
                timeout=30
            )

            if response.status_code != 200:
                raise VoiceApiError(
                    f"Failed to get recording: {response.text}",
                    response.status_code
                )

            # Save recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/tmp/recording_{call_id}_{timestamp}.wav"
            with open(file_path, "wb") as f:
                f.write(response.content)

            return file_path

        except requests.RequestException as e:
            raise VoiceApiError(f"Failed to get recording: {str(e)}")
