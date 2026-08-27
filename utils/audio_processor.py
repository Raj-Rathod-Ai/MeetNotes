import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional
import static_ffmpeg
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError
from pydub import AudioSegment

# Initialize static ffmpeg binaries
static_ffmpeg.add_paths()

FFMPEG_PATH = shutil.which("ffmpeg")
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH) if FFMPEG_PATH else None

if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH


def auto_purge_old_temp_files(max_age_seconds: int = 600) -> None:
    """
    Auto-cleans any temporary meeting files created more than 10 minutes ago.
    Prevents storage buildup on free hosting (Render).
    """
    temp_base = Path(tempfile.gettempdir())
    current_time = time.time()
    
    for pattern in ["meetnote_*", "vortex_*"]:
        for folder in temp_base.glob(pattern):
            try:
                if folder.is_dir() and (current_time - folder.stat().st_mtime > max_age_seconds):
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass


def get_cookie_file() -> Optional[str]:
    """Check for cookies.txt file or YOUTUBE_COOKIES env var to bypass bot checks."""
    local_cookie = Path("cookies.txt")
    if local_cookie.exists():
        return str(local_cookie.resolve())
        
    cookies_env = os.getenv("YOUTUBE_COOKIES")
    if cookies_env and cookies_env.strip():
        temp_cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(temp_cookie_path, "w", encoding="utf-8") as f:
            f.write(cookies_env.strip())
        return temp_cookie_path
        
    return None


def download_youtube(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download any available audio or video stream from YouTube into ephemeral storage
    and extract mono 16k WAV with ffmpeg.
    """
    auto_purge_old_temp_files()
    
    target_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="meetnote_yt_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    
    out_template = str(target_dir / "%(id)s.%(ext)s")
    proxy = os.getenv("YOUTUBE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    cookie_file = get_cookie_file()
    
    # Resilient format selector: tries audio first, falls back to any video/audio stream
    options = {
        "format": "ba/b/bestaudio/best",
        "outtmpl": out_template,
        "ffmpeg_location": FFMPEG_DIR or FFMPEG_PATH,
        "geo_bypass": True,
        "geo_bypass_country": os.getenv("GEO_BYPASS_COUNTRY", "IN"),
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "96",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
    }

    if cookie_file:
        options["cookiefile"] = cookie_file

    if proxy:
        options["proxy"] = proxy.strip()
    
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "audio")
    except DownloadError as err:
        error_msg = str(err)
        if "not made this video available in your country" in error_msg or "country" in error_msg:
            raise ValueError(
                "This video is regionally restricted and cannot be downloaded directly by Render's US cloud servers. "
                "Please upload the audio/video file directly using the 'File Upload' tab."
            )
        elif "Video unavailable" in error_msg:
            raise ValueError("This YouTube video is unavailable or deleted. Please check the link.")
        elif "Sign in" in error_msg or "age-restricted" in error_msg or "bot" in error_msg:
            raise ValueError(
                "YouTube has blocked cloud IP downloads for this video. "
                "Please upload the media file directly using the 'File Upload' tab."
            )
        else:
            raise ValueError(f"Could not access YouTube video: {error_msg}")
    except Exception as exc:
        raise ValueError(f"YouTube extraction error: {exc}")
        
    wav_file = target_dir / f"{video_id}.wav"
    if not wav_file.exists():
        candidates = list(target_dir.glob(f"*{video_id}*.wav"))
        if candidates:
            wav_file = candidates[0]
            
    return str(wav_file.resolve())


def standardize_audio(file_path: str, output_dir: Optional[str] = None) -> str:
    """Resample any media format to 16kHz mono WAV for Whisper."""
    path_obj = Path(file_path)
    target_dir = Path(output_dir) if output_dir else path_obj.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = target_dir / f"{path_obj.stem}_16k.wav"

    if str(path_obj.resolve()) == str(output_path.resolve()):
        return str(output_path)

    sound = AudioSegment.from_file(file_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(str(output_path), format="wav")
    return str(output_path)


def split_audio_chunks(wav_path: str, chunk_minutes: int = 10) -> List[str]:
    """Segment audio tracks into 10-minute slices."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_len_ms = chunk_minutes * 60 * 1000

    if len(audio) <= chunk_len_ms:
        return [wav_path]

    stem = Path(wav_path).stem
    parent = Path(wav_path).parent
    chunk_files = []

    for idx, start_ms in enumerate(range(0, len(audio), chunk_len_ms)):
        segment = audio[start_ms : start_ms + chunk_len_ms]
        part_path = parent / f"{stem}_part_{idx}.wav"
        segment.export(str(part_path), format="wav")
        chunk_files.append(str(part_path))

    return chunk_files


def process_input(source: str, chunk_minutes: int = 10, work_dir: Optional[str] = None) -> List[str]:
    """Standardized entrypoint to prepare audio tracks from URL or local path."""
    source_clean = source.strip()
    if source_clean.startswith("http://") or source_clean.startswith("https://"):
        raw_audio = download_youtube(source_clean, output_dir=work_dir)
    else:
        raw_audio = source_clean

    mono_16k = standardize_audio(raw_audio, output_dir=work_dir)
    return split_audio_chunks(mono_16k, chunk_minutes=chunk_minutes)


def cleanup_files(file_paths: List[str]) -> None:
    """Immediately remove temporary audio files from disk to keep Render storage empty."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass