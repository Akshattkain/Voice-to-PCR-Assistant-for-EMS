"""OpenAI API wrapper for GPT-4 and Whisper."""

from typing import Optional

import openai

from app.config import settings


class OpenAIClient:
    """Async OpenAI API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.default_model = default_model or settings.gpt_model

    async def chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def transcribe_audio(
        self,
        audio_data: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        language: str = "en",
        response_format: str = "verbose_json",
    ) -> dict:
        """Transcribe audio using Whisper API."""
        response = await self.client.audio.transcriptions.create(
            model=model or settings.whisper_model,
            file=(filename, audio_data),
            language=language,
            response_format=response_format,
        )
        return response
