"""
check_models.py
---------------
Lists all Gemini models available for your API key that support generateContent.
Run this to find valid model names for llm_service.py.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    print("ERROR: No GEMINI_API_KEY found in .env or environment.")
    exit(1)

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    print("\n=== Available Gemini Models (generateContent supported) ===\n")
    count = 0
    for model in client.models.list():
        name = getattr(model, "name", str(model))
        # Only show text generation models
        if "gemini" in name.lower():
            print(f"  {name}")
            count += 1
    print(f"\nTotal: {count} models\n")
except Exception as e:
    print(f"Error: {e}")
