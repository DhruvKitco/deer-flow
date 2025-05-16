# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict
import logging
import httpx

from langchain_openai import ChatOpenAI

from src.config import load_yaml_config
from src.config.agents import LLMType

# Cache for LLM instances
_llm_cache: dict[LLMType, Any] = {}

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str) -> bool:
    """Check if Ollama API is available at the given base URL."""
    try:
        # Try to connect to Ollama API
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama API not available: {e}")
        return False


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> Any:
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL"),
        "basic": conf.get("BASIC_MODEL"),
        "vision": conf.get("VISION_MODEL"),
    }
    llm_conf = llm_type_map.get(llm_type)
    if not llm_conf:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM Conf: {llm_type}")

    # Check if this is an Ollama model
    model_name = llm_conf.get("model", "")
    base_url = llm_conf.get("base_url", "")

    if "ollama" in model_name and base_url:
        # Always use ChatOpenAI for Ollama models since ChatOllama doesn't support binding tools
        logger.info(f"Using ChatOpenAI for Ollama model: {model_name}")
        # Create a copy of the configuration to avoid modifying the original
        openai_conf = llm_conf.copy()
        # If the model name contains "ollama/", remove it for ChatOpenAI
        if "ollama/" in openai_conf.get("model", ""):
            openai_conf["model"] = openai_conf["model"].replace("ollama/", "")
        return ChatOpenAI(**openai_conf)

    # Default to ChatOpenAI
    return ChatOpenAI(**llm_conf)


def get_llm_by_type(
        llm_type: LLMType,
) -> Any:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


# Initialize LLMs for different purposes - now these will be cached
basic_llm = get_llm_by_type("basic")

# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")

if __name__ == "__main__":
    print(basic_llm.invoke("Hello"))
