import os
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("voice-assistant-dev-api-key")

if not api_key:
    raise RuntimeError("API key missing from .env")

response = httpx.get(
    "https://api.elevenlabs.io/v1/user",
    headers={"xi-api-key": api_key},
    timeout=20.0,
)

print("Status:", response.status_code)

if response.is_success:
    print("ElevenLabs connection successful!")
else:
    print("Request failed:", response.text)