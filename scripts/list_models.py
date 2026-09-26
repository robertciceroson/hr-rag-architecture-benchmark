"""List the chat models your Groq key can use (never prints the key)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_experiments import load_dotenv  # noqa: E402

load_dotenv()
from groq import Groq  # noqa: E402

SKIP = ("whisper", "orpheus", "guard", "safeguard", "tts")
for m in sorted(Groq().models.list().data, key=lambda m: m.id):
    if not any(s in m.id for s in SKIP):
        print(m.id)
