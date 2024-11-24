from dataclasses import dataclass


@dataclass
class HammingCallResponseDTO:
    """DTO for responses from Hamming call initiation"""
    id: str
