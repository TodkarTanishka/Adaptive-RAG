"""
llm_service.py
--------------
Handles communication with Google's Gemini LLM:
1. Reads GEMINI_API_KEY from environment variables (.env).
2. Supports a built-in Mock Mode for offline/free testing without an API key.
3. Provides simple, clean functions to call the LLM and generate answers.
"""

import os
import time
from typing import Optional, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def is_mock_mode() -> bool:
    """
    Checks if Mock Mode is enabled.
    Returns True if MOCK_MODE is 'true' in env or if GEMINI_API_KEY is empty.
    """
    mock_env = os.getenv("MOCK_MODE", "true").lower() == "true"
    api_key_missing = not bool(os.getenv("GEMINI_API_KEY", "").strip())
    return mock_env or api_key_missing


# Primary model and supported fallback model
# These are confirmed names for the google-genai SDK (v1beta API)
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash-lite")


def _is_transient_error(error: Exception) -> bool:
    """
    Checks if an error is a temporary server condition eligible for retry.
    Catches 503 UNAVAILABLE ('high demand'), 429 (rate limits), and 500/502/504 errors.
    NEVER treats 404 NOT_FOUND or 400 BAD_REQUEST as transient.
    """
    err_str = str(error).lower()
    code = getattr(error, "code", None)

    # 404 and 400 are permanent errors — wrong model name or bad request, never retry
    if code in (400, 404):
        return False
    if "404" in err_str or "not_found" in err_str or "400" in err_str:
        return False

    if code in (429, 500, 502, 503, 504):
        return True

    transient_indicators = [
        "503", "unavailable", "high demand", "overloaded",
        "429", "resource_exhausted", "rate limit",
        "500", "internal", "temporary"
    ]
    return any(ind in err_str for ind in transient_indicators)


def _generate_helpful_unavailable_message(primary_model: str, fallback_model: Optional[str], attempts: int) -> str:
    """Returns a beginner-friendly message when Gemini service is temporarily overloaded."""
    models_tested = primary_model if not fallback_model else f"'{primary_model}' and fallback '{fallback_model}'"
    return (
        "⚠️ **Gemini Service Temporarily Unavailable (503 High Demand)**\n\n"
        f"Google's Gemini API is currently experiencing peak traffic and high demand on {models_tested} "
        f"after {attempts} retry attempts with exponential backoff.\n\n"
        "**💡 What you can do:**\n"
        "1. **Wait 30–60 seconds** and click **Rerun** or re-send your question — free-tier capacity spikes usually clear quickly.\n"
        "2. **Enable Mock Mode** using the toggle at the top of the sidebar to continue testing all features without API limits.\n"
        "3. Try asking a more specific question with shorter context if your PDF is very large."
    )


def _discover_working_model(client: Any, prompt: str) -> Optional[str]:
    """
    Auto-discovers a working Gemini model by querying the API's model list.
    Called as a last resort when both primary and fallback models return 404.
    Tries each available model that supports 'generateContent'.
    """
    # Preferred model order — fast text-only flash variants
    preferred_prefixes = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]
    try:
        available = [
            getattr(m, "name", str(m))
            for m in client.models.list()
            if "gemini" in getattr(m, "name", str(m)).lower()
        ]
    except Exception:
        return None

    # Sort by preferred order
    def sort_key(name):
        for i, prefix in enumerate(preferred_prefixes):
            if prefix in name:
                return i
        return len(preferred_prefixes)

    available.sort(key=sort_key)

    for model_name in available:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except Exception:
            continue

    return None


def call_gemini(
    prompt: str,
    api_key: Optional[str] = None,
    mock: Optional[bool] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    client_override: Optional[Any] = None,
) -> str:
    """
    Sends a prompt to the Gemini API with automatic retries, exponential backoff,
    and fallback model support for temporary 503/429 server conditions.

    Parameters:
        prompt (str): Text prompt to send to LLM.
        api_key (str, optional): User-provided API key or fallback to env.
        mock (bool, optional): Force mock mode or auto-detect.
        max_retries (int): Maximum retry attempts per model (default: 3).
        base_delay (float): Base seconds for exponential backoff (default: 1.0s).
        client_override (Any, optional): Custom/mocked client for automated tests.

    Returns:
        str: Response text from the model or mock generator.
    """
    # Use explicit API key if provided, otherwise check environment
    effective_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()

    # Determine mock status
    if mock is True or (mock is None and (not effective_key or os.getenv("MOCK_MODE", "").lower() == "true")):
        return _generate_mock_response(prompt)

    # Models to attempt in sequence
    models_to_try = [PRIMARY_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error: Optional[Exception] = None
    all_errors_transient = True

    # Attempt call via modern google-genai SDK
    try:
        from google import genai
        client = client_override if client_override is not None else genai.Client(api_key=effective_key)

        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    return response.text
                except Exception as err:
                    last_error = err
                    is_transient = _is_transient_error(err)

                    if not is_transient:
                        all_errors_transient = False
                        # 404 = wrong model name, skip to next model immediately (no retries)
                        break

                    # Transient error (503 / 429) -> wait with exponential backoff if retries left
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                    # else: retries exhausted for this model; fall through to next model

        # If all models failed, try to auto-discover a working model from the API
        if last_error is not None:
            try:
                discovered = _discover_working_model(client, prompt)
                if discovered:
                    return discovered
            except Exception:
                pass

        # If all attempts across models failed due to transient capacity
        if all_errors_transient and last_error is not None and _is_transient_error(last_error):
            return _generate_helpful_unavailable_message(PRIMARY_MODEL, FALLBACK_MODEL, max_retries)

        return (
            f"Gemini API Error: {str(last_error)}\n\n"
            "**Possible fix:** The configured model may not be available on your API plan.\n"
            "Try toggling **Mock Mode** in the sidebar to test without API limits."
        )
    except ImportError:
        pass
    except Exception as e:
        if _is_transient_error(e):
            return _generate_helpful_unavailable_message(PRIMARY_MODEL, FALLBACK_MODEL, max_retries)
        return f"Gemini API Error: {str(e)}\n\n(Tip: Toggle Mock Mode to test without an active API key.)"

    # Fallback to legacy google-generativeai SDK if modern SDK is not available
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=effective_key)
        for model_name in ["gemini-1.5-flash"]:
            for attempt in range(max_retries):
                try:
                    model = legacy_genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as err:
                    last_error = err
                    if not _is_transient_error(err):
                        raise err
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))

        if last_error and _is_transient_error(last_error):
            return _generate_helpful_unavailable_message(PRIMARY_MODEL, FALLBACK_MODEL, max_retries)
        return f"Gemini API Error: {str(last_error)}\n\n(Tip: Toggle Mock Mode to test without an active API key.)"
    except Exception as e:
        if _is_transient_error(e):
            return _generate_helpful_unavailable_message(PRIMARY_MODEL, FALLBACK_MODEL, max_retries)
        return f"Gemini API Error: {str(e)}\n\n(Tip: Run 'pip install google-genai' or enable Mock Mode.)"


def _generate_mock_response(prompt: str) -> str:
    """
    Generates realistic, beginner-friendly mock responses when running without an API key.
    """
    prompt_lower = prompt.lower()

    # Case 1: Query Classification Prompt
    if "user query:" in prompt_lower or "classify" in prompt_lower:
        # Extract the exact query inside quotes: User Query: "..."
        if 'user query: "' in prompt_lower:
            query_part = prompt_lower.split('user query: "')[1].split('"')[0]
        else:
            query_part = prompt_lower

        if any(w in query_part for w in ["what is the uploaded", "what is this pdf about", "what is the pdf about", "what is this document about", "summarize", "summary", "overview"]):
            return "DOCUMENT_SUMMARY"
        elif any(greet in query_part for greet in ["hello", "hi", "hey", "who are you", "what can you do", "weather"]):
            return "DIRECT"
        elif any(vague in query_part for vague in ["what?", "huh", "tell me", "why?"]) and len(query_part.split()) < 5:
            return "CLARIFY"
        else:
            return "DOCUMENT_QUERY"

    # Case 2: Document Summary Synthesis Prompt
    if "provide a clear overview and summary of this document" in prompt_lower or "summary of the document" in prompt_lower:
        return (
            "📑 **[MOCK MODE DOCUMENT SUMMARY]**\n\n"
            "Here is an overview of the uploaded document based on its introductory sections:\n\n"
            "- **Topic:** The document presents research and findings in its respective domain.\n"
            "- **Main Objective:** It explores key methodologies, frameworks, and experimental evaluations.\n"
            "- **Key Takeaway:** The authors demonstrate performance improvements and propose new directions.\n\n"
            "*(Simulated response in Mock Mode. Connect your Gemini API Key in the sidebar for full AI summaries!)*"
        )

    # Case 3: Document RAG Synthesis Prompt
    if "context from the document" in prompt_lower or "context:" in prompt_lower:
        return (
            "🤖 **[MOCK MODE ANSWER]**\n\n"
            "Based on the retrieved context from your document, here is the answer to your question:\n\n"
            "- The document discusses key topics related to your query.\n"
            "- Key facts and relevant details were extracted from the matched chunks.\n\n"
            "*(This simulated response is generated in Mock Mode. Connect a Gemini API Key to see real model outputs!)*"
        )

    # Case 4: Direct Answer Prompt (no retrieval needed)
    return (
        "🤖 **[MOCK MODE ANSWER]**\n\n"
        "Hello! I am your Adaptive RAG assistant running in **Mock Mode**. "
        "I was able to answer this question directly without searching through your document. "
        "Feel free to upload a PDF and ask questions about it!"
    )
