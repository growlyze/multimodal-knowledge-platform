"""
Media extractor — images, video, audio.

Returns a lightweight description. The actual visual analysis
is done by the LLM's vision capability at query time.
Permanent ingestion (vision LLM → text description → Qdrant)
happens async via the ingestion pipeline.
"""

import logging

logger = logging.getLogger("haystack-api.extract.media")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "svg"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}


async def extract_media(filename: str, ext: str, file_bytes: bytes) -> str:
    """
    Return a placeholder for media files.
    No heavy processing at upload time — vision model handles images,
    and the video pipeline handles video/audio.
    """
    size_mb = len(file_bytes) / (1024 * 1024)

    if ext in IMAGE_EXTENSIONS:
        return (
            f"Image file: {filename}\n"
            f"Size: {size_mb:.1f} MB\n"
            f"Type: {ext.upper()}\n"
            f"This image can be viewed directly by the AI vision model."
        )

    if ext in VIDEO_EXTENSIONS:
        return (
            f"Video file: {filename}\n"
            f"Size: {size_mb:.1f} MB\n"
            f"Type: {ext.upper()}\n"
            f"This video will be processed by the ingestion pipeline "
            f"for transcription and frame analysis."
        )

    if ext in AUDIO_EXTENSIONS:
        return (
            f"Audio file: {filename}\n"
            f"Size: {size_mb:.1f} MB\n"
            f"Type: {ext.upper()}\n"
            f"This audio will be transcribed by the ingestion pipeline."
        )

    return f"Media file: {filename} ({size_mb:.1f} MB)"
