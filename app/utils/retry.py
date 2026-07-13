import os
import time
import logging
from typing import Any, Callable, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger("gapnavigator.utils.retry")

class GeminiRetryLimitExceededError(Exception):
    """Exception raised when Gemini API calls fail after all retry attempts."""
    pass

def retry_gemini_call(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Executes a Gemini API function with automatic retry logic:
    - Retries up to 5 times on 429 (Resource Exhausted) and 503 (Service Unavailable) errors.
    - Uses exponential backoff: 2s, 4s, 8s, 16s, 32s.
    - Displays a Streamlit toast message while retrying.
    - Logs every retry attempt.
    - Raises GeminiRetryLimitExceededError if all retries fail.
    """
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            # Check if this is a HTTP 429 or HTTP 503 error (including name-based codes)
            is_transient = any(
                code in err_str 
                for code in ["429", "503", "RESOURCE_EXHAUSTED", "SERVICE_UNAVAILABLE", "ResourceExhausted", "ServiceUnavailable"]
            )
            
            if not is_transient:
                # If it's a 404 or other non-transient error, raise it immediately
                logger.error(f"Non-transient Gemini API error: {err_str}")
                raise e
            
            wait_time = 2 ** attempt  # 2s, 4s, 8s, 16s, 32s
            
            # Log the retry attempt
            logger.warning(
                f"Gemini API call failed (Attempt {attempt}/{max_retries}). "
                f"Retrying in {wait_time}s... Error: {err_str}"
            )
            
            # Show a user-friendly loading/retry message in Streamlit
            try:
                import streamlit as st
                if st.runtime.exists():
                    error_type = "Busy (429)" if "429" in err_str or "RESOURCE" in err_str else "Unavailable (503)"
                    st.toast(
                        f"⏳ Gemini API {error_type}. Retrying (attempt {attempt}/{max_retries}) in {wait_time}s...", 
                        icon="⚠️"
                    )
            except Exception:
                pass
            
            if attempt == max_retries:
                # Raise user-friendly message if all retries are exhausted
                raise GeminiRetryLimitExceededError(
                    "Google Gemini API is currently overloaded or experiencing high traffic. "
                    "We attempted to retry 5 times with backoff, but the request still timed out or ran out of quota. "
                    "Please wait a minute and try again."
                ) from e
            
            time.sleep(wait_time)

class RetryingChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI wrapper that automatically retries transient API errors."""
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return retry_gemini_call(super().invoke, *args, **kwargs)

class RetryingGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """GoogleGenerativeAIEmbeddings wrapper that automatically retries transient API errors."""
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return retry_gemini_call(super().embed_documents, texts, **kwargs)
        
    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        return retry_gemini_call(super().embed_query, text, **kwargs)
