"""Audio format validation and utilities."""

SUPPORTED_FORMATS = {"wav", "mp3", "webm", "m4a", "ogg", "flac"}
MAX_FILE_SIZE_MB = 25  # Whisper API limit


def validate_audio_format(filename: str) -> str:
    """Extract and validate audio format from filename. Returns format string."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return ext


def validate_audio_size(data: bytes) -> None:
    """Validate audio file size is within API limits."""
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Audio file too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB"
        )
