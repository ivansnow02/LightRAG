import asyncio
import logging
import os

import pipmaster as pm  # Pipmaster for dynamic library install

from ..utils import VERBOSE_DEBUG, verbose_debug
from langchain_core.rate_limiters import InMemoryRateLimiter
# install specific modules
if not pm.is_installed("langchain-google-genai"):
    pm.install("langchain-google-genai")

if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")

from typing import Any

import numpy as np
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import (
    locate_json_string_body_from_string,
    logger,
    safe_unicode_decode,
    wrap_embedding_func_with_attrs,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)
def create_langchain_gemini_client(
    api_key: str | None = None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.1,
    max_output_tokens: int = 1000,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Create a LangChain Google Generative AI client with the given configuration.

    Args:
        api_key: Google API key. If None, uses the GOOGLE_API_KEY environment variable.
        model_name: Model name to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro").
        temperature: Temperature for response generation.
        max_output_tokens: Maximum number of tokens in the response.
        **kwargs: Additional configuration options for the ChatGoogleGenerativeAI client.

    Returns:
        A ChatGoogleGenerativeAI client instance.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

    client_configs = {
        "model": model_name,
        "google_api_key": api_key,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        **kwargs,
    }

    return ChatGoogleGenerativeAI(**client_configs, rate_limiter=rate_limiter)


@retry(
    stop=stop_after_attempt(10),  # 增加最大重试次数
    wait=wait_exponential(
        multiplier=3, min=3, max=60
    ),  # 限流时指数退避，最小3秒，最大60秒
    retry=(
        retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(InvalidResponseError)
        | retry_if_exception_type(LangChainException)
    ),
)
async def langchain_gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> str:
    """Complete a prompt using LangChain's Google Generative AI with caching support.

    Args:
        model: The Gemini model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        api_key: Optional Google API key. If None, uses the GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
        token_tracker: Optional token tracker for usage statistics.
        **kwargs: Additional keyword arguments to pass to the LangChain client.
            Special kwargs:
            - temperature: Temperature for response generation (default: 0.1)
            - max_output_tokens: Maximum tokens in response (default: 1000)
            - hashing_kv: Will be removed from kwargs before passing to client.
            - keyword_extraction: Will be removed from kwargs before passing to client.

    Returns:
        The completed text.

    Raises:
        InvalidResponseError: If the response from Gemini is invalid or empty.
        google_exceptions.ResourceExhausted: If the Gemini API rate limit is exceeded.
        google_exceptions.DeadlineExceeded: If the Gemini API request times out.
        google_exceptions.ServiceUnavailable: If the Gemini API is unavailable.
        LangChainException: If there is an error with LangChain.
    """
    if history_messages is None:
        history_messages = []

    # Set genai logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("google.generativeai").setLevel(logging.INFO)

    # Remove special kwargs that shouldn't be passed to LangChain
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Extract client configuration options
    temperature = kwargs.pop("temperature", 0.1)
    max_output_tokens = kwargs.pop("max_output_tokens", 1000)

    # Create the LangChain Gemini client
    langchain_client = create_langchain_gemini_client(
        api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs,
    )

    # Prepare messages for LangChain
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    # Add history messages
    for msg in history_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        # Skip system messages in history as they should be handled separately

    # Add the current user prompt
    messages.append(HumanMessage(content=prompt))

    logger.debug("===== Entering func of LangChain Gemini LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LangChain Gemini LLM =====")

    try:
        # Use LangChain's invoke method
        response = await langchain_client.ainvoke(messages)

        if not response or not hasattr(response, "content"):
            logger.error("Invalid response from LangChain Gemini API")
            raise InvalidResponseError("Invalid response from LangChain Gemini API")

        content = response.content

        if not content or content.strip() == "":
            logger.error("Received empty content from LangChain Gemini API")
            raise InvalidResponseError(
                "Received empty content from LangChain Gemini API"
            )

        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))

        # Token tracking (if available in response metadata)
        if token_tracker and hasattr(response, "response_metadata"):
            usage_metadata = response.response_metadata.get("usage_metadata", {})
            if usage_metadata:
                token_counts = {
                    "prompt_tokens": usage_metadata.get("prompt_token_count", 0),
                    "completion_tokens": usage_metadata.get(
                        "candidates_token_count", 0
                    ),
                    "total_tokens": usage_metadata.get("total_token_count", 0),
                }
                token_tracker.add_usage(token_counts)

        logger.debug(f"Response content len: {len(content)}")
        verbose_debug(f"Response content: {content}")

        return content

    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Google API Rate Limit Error: {e}, will auto sleep and retry.")
        await asyncio.sleep(3)  # 主动等待3秒，防止速率过快
        raise
    except google_exceptions.DeadlineExceeded as e:
        logger.error(f"Google API Timeout Error: {e}")
        raise
    except google_exceptions.ServiceUnavailable as e:
        logger.error(f"Google API Service Unavailable Error: {e}")
        raise
    except LangChainException as e:
        logger.error(f"LangChain Error: {e}")
        raise
    except Exception as e:
        logger.error(
            f"LangChain Gemini API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise


async def langchain_gemini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using LangChain's Google Generative AI.

    Args:
        prompt: The prompt to complete.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history.
        keyword_extraction: Whether this is for keyword extraction (affects response format).
        **kwargs: Additional configuration options.

    Returns:
        The completed text.
    """
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    return await langchain_gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def langchain_gemini_1_5_flash_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using LangChain's Gemini 1.5 Flash model.

    Args:
        prompt: The prompt to complete.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history.
        keyword_extraction: Whether this is for keyword extraction.
        **kwargs: Additional configuration options.

    Returns:
        The completed text.
    """
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    result = await langchain_gemini_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

    if keyword_extraction:
        result_json = locate_json_string_body_from_string(result)
        return result_json if result_json is not None else result
    return result


async def langchain_gemini_1_5_pro_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using LangChain's Gemini 1.5 Pro model.

    Args:
        prompt: The prompt to complete.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history.
        keyword_extraction: Whether this is for keyword extraction.
        **kwargs: Additional configuration options.

    Returns:
        The completed text.
    """
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    result = await langchain_gemini_complete_if_cache(
        "gemini-1.5-pro",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

    if keyword_extraction:
        result_json = locate_json_string_body_from_string(result)
        return result_json if result_json is not None else result
    return result


async def langchain_gemini_2_0_flash_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    """Complete a prompt using LangChain's Gemini 2.0 Flash model.

    Args:
        prompt: The prompt to complete.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history.
        keyword_extraction: Whether this is for keyword extraction.
        **kwargs: Additional configuration options.

    Returns:
        The completed text.
    """
    if history_messages is None:
        history_messages = []

    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    result = await langchain_gemini_complete_if_cache(
        "gemini-2.0-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

    if keyword_extraction:
        result_json = locate_json_string_body_from_string(result)
        return result_json if result_json is not None else result
    return result


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=2048)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(LangChainException)
    ),
)
async def langchain_gemini_embed(
    texts: list[str],
    model: str = "models/embedding-001",
    api_key: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate embeddings for a list of texts using LangChain's Google Generative AI Embeddings.

    Args:
        texts: List of texts to embed.
        model: The Google embedding model to use.
        api_key: Optional Google API key. If None, uses the GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
        **kwargs: Additional configuration options.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        google_exceptions.ResourceExhausted: If the Google API rate limit is exceeded.
        google_exceptions.DeadlineExceeded: If the Google API request times out.
        google_exceptions.ServiceUnavailable: If the Google API is unavailable.
        LangChainException: If there is an error with LangChain.
    """
    if not api_key:
        api_key_value = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not api_key_value:
            raise ValueError(
                "API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        api_key = api_key_value

    # Create the LangChain Google Generative AI Embeddings client
    embeddings_client = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
        **kwargs,
    )

    try:
        # Generate embeddings using LangChain
        embeddings = await embeddings_client.aembed_documents(texts)
        return np.array(embeddings)

    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Google API Rate Limit Error: {e}")
        raise
    except google_exceptions.DeadlineExceeded as e:
        logger.error(f"Google API Timeout Error: {e}")
        raise
    except google_exceptions.ServiceUnavailable as e:
        logger.error(f"Google API Service Unavailable Error: {e}")
        raise
    except LangChainException as e:
        logger.error(f"LangChain Embeddings Error: {e}")
        raise
    except Exception as e:
        logger.error(f"LangChain Gemini Embeddings Call Failed: {e}")
        raise
