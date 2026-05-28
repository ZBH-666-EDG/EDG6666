"""AI analysis configuration. Loads API keys from environment or .env file."""
import os


def _load_dotenv():
    """Load .env file if python-dotenv is available."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.isfile(env_path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


_load_dotenv()

# ---- AI Model Configuration ----
AI_PROVIDER = os.getenv('AI_PROVIDER', 'deepseek')  # 'deepseek' or 'gemini'
AI_API_KEY = os.getenv('AI_API_KEY', '')
AI_MODEL = os.getenv('AI_MODEL', 'deepseek-chat')    # deepseek-chat supports vision
AI_TIMEOUT = int(os.getenv('AI_TIMEOUT', '10'))       # seconds
AI_ENABLED = bool(AI_API_KEY)

# ---- API Endpoints ----
AI_ENDPOINTS = {
    'deepseek': 'https://api.deepseek.com/v1/chat/completions',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
}


def get_endpoint():
    base = AI_ENDPOINTS.get(AI_PROVIDER, AI_ENDPOINTS['deepseek'])
    if AI_PROVIDER == 'gemini':
        return base.format(model=AI_MODEL) + '?key=' + AI_API_KEY
    return base
