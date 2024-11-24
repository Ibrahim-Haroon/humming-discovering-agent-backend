from dataclasses import dataclass


@dataclass
class HammingCallRequestDTO:
    """DTO for initiating calls via the Hamming API"""
    phone_number: str
    prompt: str
    webhook_url: str
