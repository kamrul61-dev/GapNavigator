import pytest
import time
from unittest.mock import MagicMock, patch
from app.utils.retry import (
    retry_gemini_call, 
    GeminiRetryLimitExceededError, 
    RetryingChatGoogleGenerativeAI,
    RetryingGoogleGenerativeAIEmbeddings
)

def test_retry_success_first_attempt():
    """Verify that a successful call on the first attempt does not retry and returns the result."""
    mock_func = MagicMock(return_value="success_result")
    
    result = retry_gemini_call(mock_func)
    
    assert result == "success_result"
    assert mock_func.call_count == 1

@patch("time.sleep") # Mock sleep to run instantly
def test_retry_on_transient_error_resolves(mock_sleep):
    """Verify retry handles a transient 429 error and succeeds on a subsequent try."""
    # Raise 429 twice, then succeed
    side_effects = [
        Exception("RESOURCE_EXHAUSTED: Quota exceeded (HTTP 429)"),
        Exception("Service Unavailable (HTTP 503)"),
        "success_result"
    ]
    mock_func = MagicMock(side_effect=side_effects)
    
    result = retry_gemini_call(mock_func)
    
    assert result == "success_result"
    assert mock_func.call_count == 3
    # Check that backoff wait times were correct: 2 ** 1 = 2 seconds, 2 ** 2 = 4 seconds
    mock_sleep.assert_any_call(2)
    mock_sleep.assert_any_call(4)
    assert mock_sleep.call_count == 2

def test_non_transient_error_fails_immediately():
    """Verify that non-transient errors (e.g. 404 NOT_FOUND) fail immediately without retrying."""
    mock_func = MagicMock(side_effect=Exception("404 NOT_FOUND: model not found"))
    
    with pytest.raises(Exception) as exc_info:
        retry_gemini_call(mock_func)
        
    assert "404" in str(exc_info.value)
    assert mock_func.call_count == 1  # Should not retry

@patch("time.sleep")
def test_exhausted_retries_raises_user_friendly_error(mock_sleep):
    """Verify that when all 5 retries fail with 429, a user-friendly error is raised."""
    mock_func = MagicMock(side_effect=Exception("RESOURCE_EXHAUSTED: Quota exceeded (HTTP 429)"))
    
    with pytest.raises(GeminiRetryLimitExceededError) as exc_info:
        retry_gemini_call(mock_func)
        
    assert "Google Gemini API is currently overloaded" in str(exc_info.value)
    assert mock_func.call_count == 5  # Initial + 4 retries (Wait, range 1 to 5 means 5 attempts in total)
    assert mock_sleep.call_count == 4  # 4 sleep delays between 5 attempts
    mock_sleep.assert_any_call(2)
    mock_sleep.assert_any_call(4)
    mock_sleep.assert_any_call(8)
    mock_sleep.assert_any_call(16)
