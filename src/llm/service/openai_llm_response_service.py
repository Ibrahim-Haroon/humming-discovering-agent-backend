import requests
from typing import override
from src.util.env import Env
from src.llm.service.llm_response_service import LlmResponseService


class OpenAILlmResponseService(LlmResponseService):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.__model = model
        self.__url = "https://api.openai.com/v1/chat/completions"
        self.__api_key = Env()["OPENAI_API_KEY"]
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}",
        }

    @property
    def model(self) -> str:
        return self.__model

    @override
    def response(self, role: str, prompt: str) -> str:
        payload = {
            "model": self.__model,
            "messages": [
                {
                    "role": "system",
                    "content": role
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            url=self.__url,
            headers=self.__headers,
            json=payload
        )

        response_data = response.json()
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError("No content found in the response")
