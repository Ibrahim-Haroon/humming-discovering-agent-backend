import time
import requests
from typing import Optional
from datetime import datetime
from src.util.env import Env
from src.rest.dto.hamming_call_request_dto import HammingCallRequestDTO
from src.rest.dto.hamming_call_response_dto import HammingCallResponseDTO
from src.rest.dto.hamming_webhook_response_dto import HammingWebhookResponseDTO


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

    def start_call(
        self,
        phone_number: str,
        prompt: str,
        webhook_url: Optional[str] = None
    ) -> HammingCallResponseDTO:
        """
        Initiates a new phone call

        :param phone_number: Target phone number
        :param prompt: System prompt for the agent
        :param webhook_url: Optional webhook URL for status updates
        :return: Call response containing call ID
        :raises VoiceApiError: If API call fails
        """
        try:
            request = HammingCallRequestDTO(
                phone_number=phone_number,
                prompt=prompt,
                webhook_url=webhook_url or ""
            )

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
            return HammingCallResponseDTO(id=data["id"])

        except requests.RequestException as e:
            raise VoiceApiError(f"API request failed: {str(e)}")

    def get_recording(
        self,
        call_id: str,
        max_retries: int = 3,
        retry_delay: int = 30
    ) -> str:
        """
        Gets the recording for a completed call

        :param call_id: ID of the call
        :param max_retries: Maximum number of retry attempts
        :param retry_delay: Delay between retries in seconds
        :return: Path to downloaded recording
        :raises VoiceApiError: If recording retrieval fails
        """
        attempts = 0

        while attempts < max_retries:
            try:
                response = requests.get(
                    f"{self.__base_url}/media/exercise",
                    headers=self.__headers,
                    params={"id": call_id},
                    timeout=30
                )

                if response.status_code == 404:
                    # Recording not ready yet
                    attempts += 1
                    if attempts < max_retries:
                        time.sleep(retry_delay)
                        continue
                    raise VoiceApiError("Recording not available after retries")

                if response.status_code != 200:
                    raise VoiceApiError(
                        f"Failed to get recording: {response.text}",
                        response.status_code
                    )

                # Save recording to temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"/tmp/recording_{call_id}_{timestamp}.wav"

                with open(file_path, "wb") as f:
                    f.write(response.content)

                return file_path

            except requests.RequestException as e:
                attempts += 1
                if attempts >= max_retries:
                    raise VoiceApiError(f"Failed to get recording after retries: {str(e)}")
                time.sleep(retry_delay)

    @staticmethod
    def process_webhook(payload: dict) -> HammingWebhookResponseDTO:
        """
        Processes webhook callbacks from the API

        :param payload: Webhook payload from API
        :return: Processed webhook response
        :raises ValueError: If payload is invalid
        """
        try:
            return HammingWebhookResponseDTO(
                id=payload["id"],
                status=payload["status"],
                recording_available=payload["recording_available"]
            )
        except KeyError as e:
            raise ValueError(f"Invalid webhook payload: missing {e}")
