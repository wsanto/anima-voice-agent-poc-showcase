"""
Custom ElevenLabs V3 TTS Service for Pipecat
Uses HTTP REST API instead of WebSocket for v3 model compatibility
"""

import asyncio
import aiohttp
import json
import os
import io
from typing import AsyncGenerator, Optional
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language
from loguru import logger

try:
    import pydub
    from pydub import AudioSegment
except ImportError:
    logger.error("pydub is required for audio conversion. Install with: pip install pydub")
    raise


class ElevenLabsV3TTSService(TTSService):
    """ElevenLabs V3 TTS Service using HTTP REST API"""
    
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_v3",
        stability: float = 0.5,
        similarity_boost: float = 0.8,
        speed: float = 1.0,
        sample_rate: int = 24000,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._stability = stability
        self._similarity_boost = similarity_boost
        self._speed = speed
        self._base_url = "https://api.elevenlabs.io/v1"
        
        # Validate v3 stability values
        if model == "eleven_v3" and stability not in [0.0, 0.5, 1.0]:
            logger.warning(f"V3 model requires stability to be 0.0, 0.5, or 1.0. Adjusting {stability} to 0.5")
            self._stability = 0.5
            
        logger.info(f"ElevenLabs V3 TTS initialized with model: {model}, voice: {voice_id}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio using ElevenLabs V3 REST API"""
        logger.debug(f"ElevenLabs V3 TTS generating: {text[:50]}...")
        
        yield TTSStartedFrame()
        
        try:
            url = f"{self._base_url}/text-to-speech/{self._voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self._api_key
            }
            
            data = {
                "text": text,
                "model_id": self._model,
                "voice_settings": {
                    "stability": self._stability,
                    "similarity_boost": self._similarity_boost
                }
            }
            
            # Add speed for non-v3 models
            if self._model != "eleven_v3":
                data["voice_settings"]["speed"] = self._speed
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error {response.status}: {error_text}")
                        yield ErrorFrame(f"ElevenLabs API error: {response.status}")
                        return
                    
                    # Read audio data
                    audio_data = await response.read()
                    
                    if not audio_data:
                        logger.error("No audio data received from ElevenLabs")
                        yield ErrorFrame("No audio data received")
                        return
                    
                    # Convert MP3 to raw PCM audio
                    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                    
                    # Convert to the target sample rate and mono
                    audio_segment = audio_segment.set_frame_rate(self._sample_rate)
                    audio_segment = audio_segment.set_channels(1)
                    
                    # Convert to raw PCM bytes
                    raw_audio = audio_segment.raw_data
                    
                    # Split into chunks for streaming
                    chunk_size = 1024 * 2  # 2KB chunks
                    for i in range(0, len(raw_audio), chunk_size):
                        chunk = raw_audio[i:i + chunk_size]
                        if chunk:  # Only yield non-empty chunks
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=self._sample_rate,
                                num_channels=1
                            )
                    
                    logger.debug(f"Generated {len(audio_data)} bytes of audio")
                    
        except Exception as e:
            logger.error(f"ElevenLabs V3 TTS error: {e}")
            yield ErrorFrame(f"TTS generation failed: {e}")
        
        yield TTSStoppedFrame()

    async def flush_audio(self):
        """Flush any pending audio - no-op for HTTP-based service"""
        pass
