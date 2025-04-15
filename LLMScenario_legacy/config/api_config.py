import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GPT4_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"

# Configuration
MAX_TOKENS = 4096
TEMPERATURE = 0.7