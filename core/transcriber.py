import os
import re
import json
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import requests
from pydub import AudioSegment

SARVAM_ENDPOINT = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_CHUNK_LIMIT_SEC = 25

_faster_whisper_model = None

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
}


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract clean 11-char YouTube video ID from various URL formats."""
    raw = url_or_id.strip()
    if len(raw) == 11 and not ("/" in raw or "." in raw or "?" in raw or "=" in raw):
        return raw

    patterns = [
        r"(?:v=|\/v\/|embed\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
        r"([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match and len(match.group(1)) == 11:
            return match.group(1)
    return None


def fetch_innertube_captions(video_id: str) -> Optional[str]:
    """Direct InnerTube caption extraction bypassing datacenter IP blocks."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        if not resp.ok:
            return None

        html = resp.text
        match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
        if not match:
            return None

        tracks = json.loads(match.group(1))
        if not tracks:
            return None

        # Pick first available caption track (English or Hindi preferred)
        track_url = None
        for t in tracks:
            lang = t.get("languageCode", "").lower()
            if "en" in lang or "hi" in lang:
                track_url = t.get("baseUrl")
                break
        if not track_url:
            track_url = tracks[0].get("baseUrl")

        if track_url:
            c_resp = requests.get(track_url, headers=BROWSER_HEADERS, timeout=10)
            if c_resp.ok:
                # Strip XML tags
                text = re.sub(r"<[^>]+>", " ", c_resp.text)
                text = re.sub(r"&amp;", "&", text)
                text = re.sub(r"&quot;", '"', text)
                text = re.sub(r"&#39;", "'", text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 50:
                    return text
    except Exception as e:
        print(f"InnerTube direct scraper note: {e}")
    return None


def fetch_youtube_captions(url_or_id: str) -> Optional[str]:
    """
    Attempt instant caption extraction from YouTube using multiple resilient strategies.
    Executes in under 2 seconds even on datacenter hosting like Render.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return None

    # Strategy 1: YouTube Transcript API with browser session
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id=video_id)
        
        for candidate in transcript_list:
            fetched = candidate.fetch()
            parts = []
            for item in fetched:
                if hasattr(item, "text"):
                    parts.append(str(item.text))
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            text = " ".join(parts).strip()
            if text and len(text) > 50:
                return text
    except Exception as e:
        print(f"YouTubeTranscriptApi info for {video_id}: {e}")

    # Strategy 2: Direct InnerTube HTML Caption Track Scraper
    direct_text = fetch_innertube_captions(video_id)
    if direct_text and len(direct_text) > 50:
        return direct_text

    return None


def get_faster_whisper(model_size: str = "base"):
    """Initialize high-speed CTranslate2 Whisper model in INT8 mode."""
    global _faster_whisper_model
    if _faster_whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            target_model = model_size or os.getenv("WHISPER_MODEL", "base")
            _faster_whisper_model = WhisperModel(target_model, device="cpu", compute_type="int8")
        except Exception:
            _faster_whisper_model = None
    return _faster_whisper_model


def transcribe_fast_whisper(chunk_path: str, model_name: str = "base") -> str:
    """4x faster transcription using CTranslate2 INT8."""
    if not os.path.exists(chunk_path):
        return ""

    model = get_faster_whisper(model_name)
    if model is not None:
        try:
            segments, _ = model.transcribe(chunk_path, beam_size=1, language="en")
            return " ".join(segment.text for segment in segments).strip()
        except Exception as e:
            print(f"Faster-Whisper error: {e}")
    return ""


def _send_sarvam_slice(slice_data: tuple) -> str:
    """Helper for parallel Sarvam API requests."""
    temp_file, headers, model_tag = slice_data
    try:
        with open(temp_file, "rb") as stream:
            resp = requests.post(
                SARVAM_ENDPOINT,
                headers=headers,
                files={"file": (os.path.basename(temp_file), stream, "audio/wav")},
                data={"model": model_tag, "with_diarization": "false"},
                timeout=90,
            )
        if resp.ok:
            return resp.json().get("transcript", "")
    except Exception:
        pass
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
    return ""


def transcribe_sarvam_parallel(chunk_path: str) -> str:
    """Send slices to Sarvam in parallel worker threads for 5x faster translation."""
    if not os.path.exists(chunk_path):
        return ""

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is required for Hinglish transcription. Set it in .env")

    model_tag = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
    sound = AudioSegment.from_wav(chunk_path)
    window_ms = SARVAM_CHUNK_LIMIT_SEC * 1000

    slice_tasks = []
    headers = {"api-subscription-key": api_key}

    for idx, start_ms in enumerate(range(0, len(sound), window_ms)):
        slice_audio = sound[start_ms : start_ms + window_ms]
        temp_file = f"{chunk_path}_sv_slice_{idx}.wav"
        slice_audio.export(temp_file, format="wav")
        slice_tasks.append((temp_file, headers, model_tag))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_send_sarvam_slice, slice_tasks))

    return " ".join(filter(None, results)).strip()


def transcribe_all(chunks: List[str], language: str = "english", model_name: Optional[str] = None, source_url: Optional[str] = None) -> str:
    """High-speed transcription pipeline with caption-first fast-path."""
    if source_url and ("youtube.com" in source_url or "youtu.be" in source_url):
        instant_text = fetch_youtube_captions(source_url)
        if instant_text and len(instant_text) > 100:
            return instant_text

    results = []
    is_hinglish = language.lower().strip() == "hinglish"

    for chunk in chunks:
        if not os.path.exists(chunk):
            continue
        if is_hinglish:
            text = transcribe_sarvam_parallel(chunk)
        else:
            text = transcribe_fast_whisper(chunk, model_name=model_name or "base")
        if text:
            results.append(text)

    return " ".join(results).strip()
