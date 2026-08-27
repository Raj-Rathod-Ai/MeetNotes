import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional
import static_ffmpeg
import yt_dlp

# Initialize static ffmpeg binaries if system ffmpeg is missing
static_ffmpeg.add_paths()

FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_EXE = shutil.which("ffprobe") or "ffprobe"


def auto_purge_old_temp_files(max_age_seconds: int = 600) -> None:
    """
    Auto-cleans any temporary meeting files created more than 10 minutes ago.
    Prevents storage buildup on free hosting (Streamlit Cloud).
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
    """Check for cookies.txt file or YOUTUBE_COOKIES env var."""
    local_cookie = Path("cookies.txt")
    if local_cookie.exists() and local_cookie.stat().st_size > 50:
        return str(local_cookie.resolve())
        
    cookies_env = os.getenv("YOUTUBE_COOKIES")
    if cookies_env and cookies_env.strip():
        temp_cookie_path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(temp_cookie_path, "w", encoding="utf-8") as f:
            f.write(cookies_env.strip())
        return temp_cookie_path
        
    return None


def _download_stream(url: str, target_dir: Path, out_template: str, use_cookie: bool = True) -> str:
    """Attempt download with specified cookie configuration."""
    proxy = os.getenv("YOUTUBE_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    cookie_file = get_cookie_file() if use_cookie else None

    options = {
        "format": "ba/b/bestaudio/best",
        "outtmpl": out_template,
        "ffmpeg_location": FFMPEG_EXE,
        "geo_bypass": True,
        "geo_bypass_country": os.getenv("GEO_BYPASS_COUNTRY", "IN"),
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb"],
                "player_skip": ["webpage", "configs"]
            }
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
    }

    if cookie_file:
        options["cookiefile"] = cookie_file

    if proxy:
        options["proxy"] = proxy.strip()

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")

    wav_file = target_dir / f"{video_id}.wav"
    if not wav_file.exists():
        candidates = list(target_dir.glob(f"*{video_id}*.wav"))
        if candidates:
            wav_file = candidates[0]

    return str(wav_file.resolve())


def download_youtube(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download lowest bandwidth audio-only stream from YouTube into ephemeral storage
    with automatic cookie fallback and multi-client rotation.
    """
    auto_purge_old_temp_files()
    
    target_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="meetnote_yt_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    
    out_template = str(target_dir / "%(id)s.%(ext)s")
    
    # Attempt 1: Standard download with mobile client rotation
    try:
        return _download_stream(url, target_dir, out_template, use_cookie=False)
    except Exception as first_err:
        first_msg = str(first_err)
        
        # Attempt 2: If failed and cookies exist, try with cookie
        if get_cookie_file():
            try:
                return _download_stream(url, target_dir, out_template, use_cookie=True)
            except Exception:
                pass

        if "not made this video available in your country" in first_msg or "country" in first_msg:
            raise ValueError(
                "This video is regionally restricted and cannot be downloaded directly by cloud servers. "
                "Please upload the audio/video file directly using the 'File Upload' tab."
            )
        elif "Video unavailable" in first_msg:
            raise ValueError("This YouTube video is unavailable or deleted. Please check the link.")
        elif "Sign in" in first_msg or "age-restricted" in first_msg or "bot" in first_msg or "reloaded" in first_msg:
            raise ValueError(
                "YouTube requires account login or verification on cloud datacenter IPs. "
                "Please upload the recording directly using the File Upload tab."
            )
        else:
            raise ValueError(f"Could not access YouTube video: {first_msg}")


def standardize_audio(file_path: str, output_dir: Optional[str] = None) -> str:
    """Resample any media format to 16kHz mono WAV for Whisper using direct FFmpeg."""
    path_obj = Path(file_path)
    target_dir = Path(output_dir) if output_dir else path_obj.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = target_dir / f"{path_obj.stem}_16k.wav"

    if str(path_obj.resolve()) == str(output_path.resolve()):
        return str(output_path)

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(path_obj.resolve()),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path.resolve())
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(output_path)


def get_audio_duration_seconds(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        FFPROBE_EXE,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def split_audio_chunks(wav_path: str, chunk_minutes: int = 10) -> List[str]:
    """Segment audio tracks into 10-minute slices using direct FFmpeg."""
    duration = get_audio_duration_seconds(wav_path)
    chunk_len_sec = chunk_minutes * 60

    if duration <= chunk_len_sec or duration <= 0:
        return [wav_path]

    stem = Path(wav_path).stem
    parent = Path(wav_path).parent
    chunk_files = []

    num_chunks = int(duration // chunk_len_sec) + (1 if duration % chunk_len_sec > 0 else 0)
    for idx in range(num_chunks):
        start_sec = idx * chunk_len_sec
        part_path = parent / f"{stem}_part_{idx}.wav"
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-ss", str(start_sec),
            "-t", str(chunk_len_sec),
            "-i", wav_path,
            "-c", "copy",
            str(part_path.resolve())
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        chunk_files.append(str(part_path))

    return chunk_files if chunk_files else [wav_path]


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
    """Immediately remove temporary audio files from disk to keep storage empty."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass