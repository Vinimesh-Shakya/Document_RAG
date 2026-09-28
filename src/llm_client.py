import logging
from typing import Optional
from groq import Groq
from config import settings

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Base exception class for LLM client errors."""
    pass

class MissingAPIKeyException(LLMClientError):
    """Raised when the Groq API key is missing."""
    pass

class GroqAPIException(LLMClientError):
    """Raised when Groq API returns an error."""
    pass

def get_groq_client() -> Groq:
    """
    Creates and returns a Groq client instance.
    
    Returns:
        Groq: Initialized Groq client.
        
    Raises:
        MissingAPIKeyException: If GROQ_API_KEY is not configured.
    """
    if not settings.GROQ_API_KEY:
        raise MissingAPIKeyException(
            "Groq API Key is not set. Please create a '.env' file "
            "with GROQ_API_KEY=your_api_key or set it in your environment."
        )
    return Groq(api_key=settings.GROQ_API_KEY)

def generate_answer(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None
) -> str:
    """
    Sends the system prompt (rules + context) and user prompt (question) to Groq LLM.
    Uses temperature=0.0 to make the answer factual and deterministic.
    
    Args:
        system_prompt (str): The high-level instructions containing context and grounding rules.
        user_prompt (str): The user's question.
        model_name (str, optional): The name of the Groq model. Defaults to settings LLM_MODEL_NAME.
        
    Returns:
        str: The generated answer from the LLM.
        
    Raises:
        GroqAPIException: If API call fails.
    """
    if model_name is None:
        model_name = settings.LLM_MODEL_NAME

    logger.info(f"Sending request to Groq LLM using model '{model_name}'...")
    
    try:
        client = get_groq_client()
        
        # Call Groq API Chat Completion
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model_name,
            # We set temperature to 0.0 to ensure the model responds as factually as possible,
            # picking high-probability words without creative randomness.
            temperature=0.0,
            max_tokens=1024,
        )
        
        answer = completion.choices[0].message.content
        logger.info("LLM successfully generated response.")
        return answer
        
    except MissingAPIKeyException as e:
        logger.error("Missing API key error: " + str(e))
        raise e
    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")
        raise GroqAPIException(
            f"Failed to communicate with Groq LLM: {str(e)}"
        ) from e
