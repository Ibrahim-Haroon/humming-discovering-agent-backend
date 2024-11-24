from dataclasses import dataclass


@dataclass
class HammingWebhookResponseDTO:
    """DTO for webhook responses from the Hamming API"""
    id: str
    status: str
    recording_available: bool
