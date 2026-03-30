"""Anthropic API wrapper for Claude."""

from typing import Optional

import anthropic

from app.config import settings


class AnthropicClient:
    """Async Anthropic API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key or ""
        )
        self.default_model = default_model or settings.claude_model

    async def chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Send a message to Claude and return the response text."""
        response = await self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
