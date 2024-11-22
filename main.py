"""
Lyrics Transcriber
-----------------
A powerful tool to transcribe song lyrics with precise timestamps using OpenAI's Whisper model.

This script provides detailed transcription of audio files, including:
- Verse by verse analysis
- Word by word timing
- Confidence scores
- Automatic file selection
- GPU acceleration support
- Export functionality

Author: Nojs Nojin (@NojinNojs)
GitHub: https://github.com/NojinNojs
License: MIT
Copyright (c) 2024 Nojs Nojin
"""

import whisper
import torch
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
import sys
from datetime import datetime
from tqdm import tqdm
import time
import warnings
import torch.cuda
import logging
from pydub import AudioSegment
from rich.prompt import Prompt
import glob
from rich.style import Style
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.box import DOUBLE, ROUNDED

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# Menyembunyikan warning yang tidak penting
warnings.filterwarnings("ignore")

def download_with_progress(url: str, root: str, in_memory: bool) -> bytes:
    """
    Download files with progress tracking and resume capability.
    
    Args:
        url (str): URL of the file to download
        root (str): Directory to save the downloaded file
        in_memory (bool): Whether to return the file content in memory
        
    Returns:
        bytes: File path or file content depending on in_memory parameter
        
    Raises:
        Exception: If download fails or is interrupted
    """
    os.makedirs(root, exist_ok=True)
    local_path = os.path.join(root, os.path.basename(url))
    
    # Check if file already exists
    if os.path.exists(local_path):
        expected_size = int(requests.head(url).headers.get('content-length', 0))
        current_size = os.path.getsize(local_path)
        
        if current_size == expected_size:
            console.print("[green]Model already downloaded, using cached file.[/green]")
            return local_path if not in_memory else open(local_path, 'rb').read()
        elif current_size < expected_size:
            headers = {'Range': f'bytes={current_size}-'}
            mode = 'ab'
            initial_pos = current_size
            console.print(f"[yellow]Resuming download from {current_size/1024/1024:.1f}MB[/yellow]")
        else:
            mode = 'wb'
            initial_pos = 0
            headers = {}
    else:
        mode = 'wb'
        initial_pos = 0
        headers = {}
    
    try:
        # Optimize connection settings
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount('https://', adapter)
        
        # Add headers to optimize connection
        headers.update({
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=600',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        response = session.get(
            url, 
            stream=True, 
            timeout=15,
            headers=headers,
            verify=True
        )
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if initial_pos:
            total_size += initial_pos
            
        # Increase block size for faster downloads
        block_size = 8 * 1024 * 1024  # 8MB blocks

        with open(local_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_pos,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading model",
                ncols=80,
                miniters=1
            ) as pbar:
                try:
                    # Use larger chunks for faster processing
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            f.flush()  # Ensure data is written to disk
                            
                except KeyboardInterrupt:
                    f.flush()
                    console.print("\n[yellow]Download paused. Progress saved.[/yellow]")
                    console.print("[cyan]Run the script again to resume download.[/cyan]")
                    sys.exit(1)
                    
        if os.path.getsize(local_path) == total_size:
            console.print("[green]Download completed successfully![/green]")
            return local_path if not in_memory else open(local_path, 'rb').read()
        else:
            console.print("[yellow]Download incomplete, will resume on next run.[/yellow]")
            sys.exit(1)

    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        console.print(f"[bold red]Download error:[/bold red] {str(e)}")
        console.print("[yellow]Will retry on next run.[/yellow]")
        sys.exit(1)

# Override whisper's download function
whisper._download = download_with_progress

def get_audio_files() -> list:
    """
    Scan current directory for supported audio files.
    
    Supported formats:
    - MP3 (.mp3)
    - WAV (.wav)
    - M4A (.m4a)
    - FLAC (.flac)
    
    Returns:
        list: List of audio files found in current directory
    """
    extensions = ['.mp3', '.wav', '.m4a', '.flac']
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(f'*{ext}'))
    return audio_files

def select_audio_file() -> tuple:
    """
    Display interactive menu for audio file selection.
    
    Returns:
        tuple: Selected file path and AudioSegment object
        
    Raises:
        SystemExit: If no audio files found
        ValueError: If invalid selection made
    """
    audio_files = get_audio_files()
    
    if not audio_files:
        console.print("[bold red]No audio files found![/bold red]")
        console.print("[yellow]Please add audio files (mp3/wav/m4a/flac) to the current directory.[/yellow]")
        sys.exit(1)
    
    console.print("\n[bold blue]═══ 🎵 Available Audio Files ═══[/bold blue]")
    for i, file in enumerate(audio_files, 1):
        try:
            audio = AudioSegment.from_file(file)
            duration = len(audio)/1000
            console.print(f"[cyan]{i}.[/cyan] {file} [dim]({duration:.2f}s)[/dim]")
        except:
            console.print(f"[cyan]{i}.[/cyan] {file} [red](Unable to read duration)[/red]")
    
    while True:
        try:
            choice = Prompt.ask(
                "\n[bold cyan]Select audio file number[/bold cyan]",
                choices=[str(i) for i in range(1, len(audio_files) + 1)],
                show_choices=True
            )
            
            selected_file = audio_files[int(choice) - 1]
            audio = AudioSegment.from_file(selected_file)
            return selected_file, audio
            
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
        except Exception as e:
            console.print(f"[red]Error reading {selected_file}: {str(e)}[/red]")
            console.print("[yellow]Please select another file.[/yellow]")

def show_copyright():
    """Display copyright information and ASCII art banner."""
    title = Text("🎵 Lyrics Transcriber", style="bold blue")
    version = Text("v1.0.0", style="cyan")
    
    header = Table(show_header=False, box=DOUBLE, border_style="blue")
    header.add_column()
    header.add_row(Align.center(title))
    header.add_row(Align.center(version))
    
    console.print("\n")
    console.print(Align.center(header))
    
    # Subtitle and info
    subtitle = Table(show_header=False, box=None, padding=1)
    subtitle.add_column(style="dim")
    subtitle.add_row("Copyright © 2024 Nojs Nojin. All rights reserved.")
    subtitle.add_row("This software is licensed under the MIT License.")
    subtitle.add_row("GitHub: https://github.com/NojinNojs")
    
    console.print(Align.center(subtitle))
    console.print("\n" + "═" * 60 + "\n")

def show_audio_info(selected_file: str, audio: AudioSegment):
    """Display audio file information in a beautiful table."""
    info_table = Table(box=ROUNDED, border_style="cyan", show_header=False)
    info_table.add_column("Property", style="bold cyan")
    info_table.add_column("Value", style="yellow")
    
    info_table.add_row("File", selected_file)
    info_table.add_row("Sample Rate", f"{audio.frame_rate}Hz")
    info_table.add_row("Channels", str(audio.channels))
    info_table.add_row("Duration", f"{len(audio)/1000:.2f}s")
    
    console.print("\n[bold cyan]Audio Information[/bold cyan]")
    console.print(info_table)

def show_gpu_info():
    """Display GPU information in a beautiful table."""
    if torch.cuda.is_available():
        gpu_table = Table(box=ROUNDED, border_style="green", show_header=False)
        gpu_table.add_column("Property", style="bold cyan")
        gpu_table.add_column("Value", style="yellow")
        
        gpu_table.add_row("GPU", torch.cuda.get_device_name(0))
        gpu_table.add_row("CUDA Version", torch.version.cuda)
        
        console.print("\n[bold green]GPU Information[/bold green]")
        console.print(gpu_table)
    else:
        console.print("\n[bold red]⚠ GPU not detected, using CPU instead[/bold red]")

# Main execution block
if __name__ == "__main__":
    try:
        show_copyright()
        show_gpu_info()
        
        # Select and check audio file
        selected_file, audio = select_audio_file()
        show_audio_info(selected_file, audio)

        # Load model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Loading Whisper model...", total=None)
            model = whisper.load_model("large").to("cuda" if torch.cuda.is_available() else "cpu")
            progress.update(task, visible=False)

        console.print("\n[bold green]✓ Model loaded successfully![/bold green]")

        # Transcribe
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task("[cyan]Transcribing audio...", total=None)
            result = model.transcribe(
                selected_file,
                language="en",
                word_timestamps=True,
                initial_prompt="This is a song lyrics transcription with clear timestamps.",
                temperature=0.2,
                best_of=5,
                beam_size=5,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.3,
                fp16=torch.cuda.is_available()
            )

        # Process results
        segments = result["segments"]
        
        # 1. Output per verse
        console.print("\n[bold blue]═══ 🎵 Lyrics by Verse ═══[/bold blue]")
        for i, segment in enumerate(segments, 1):
            start = round(segment["start"], 2)
            end = round(segment["end"], 2)
            duration = round(end - start, 2)
            text = segment["text"].strip()
            confidence = segment.get("confidence", 0)
            
            verse_panel = Panel(
                f"""[bold white]Verse {i}[/bold white]
[cyan]Time: [{start:06.2f} → {end:06.2f}] ({duration:04.2f}s)[/cyan]
[yellow]Lyrics:[/yellow] {text}
[dim]Confidence: {confidence:.2%}[/dim]""",
                box=ROUNDED,
                border_style="blue",
                padding=(1, 2)
            )
            
            console.print(verse_panel)
            console.print("[dim]" + "─" * 50 + "[/dim]")

        # 2. Output per kata (word-by-word)
        console.print("\n[bold blue]═══ 🎵 Word-by-Word Analysis ═══[/bold blue]")
        
        all_words = []
        for segment in segments:
            if "words" in segment and segment["words"]:
                all_words.extend([
                    word for word in segment["words"]
                    if len(word.get("word", "").strip()) > 0
                ])

        # Group words into lines (based on timing gaps)
        MAX_GAP = 2.0  # seconds
        lines = []
        current_line = []
        
        for i, word in enumerate(all_words):
            current_line.append(word)
            if i < len(all_words) - 1:
                current_gap = all_words[i + 1]["start"] - word["end"]
                if current_gap > MAX_GAP:
                    lines.append(current_line)
                    current_line = []
        if current_line:
            lines.append(current_line)

        # Print detailed word analysis
        for i, line in enumerate(lines, 1):
            start = round(line[0]["start"], 2)
            end = round(line[-1]["end"], 2)
            duration = round(end - start, 2)
            
            words_detail = "\n".join([
                f"  └─ {word['word'].strip():20} [{word['start']:06.2f} → {word['end']:06.2f}]"
                for word in line
            ])
            
            line_text = " ".join([word["word"].strip() for word in line])
            
            line_panel = f"""[bold white]Line {i}[/bold white]
[cyan]Time: [{start:06.2f} → {end:06.2f}] ({duration:04.2f}s)[/cyan]
[yellow]Full line:[/yellow] {line_text}

[dim]Word-by-word timing:[/dim]
{words_detail}"""
            
            console.print(Panel(line_panel, border_style="blue", padding=(1, 2)))
            console.print("[dim]" + "─" * 50 + "[/dim]")

        # 3. Export to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"lyrics_export_{os.path.splitext(selected_file)[0]}_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== SONG LYRICS ANALYSIS ===\n\n")
            f.write(f"Audio: {selected_file}\n")
            f.write(f"Duration: {len(audio)/1000:.2f}s\n")
            f.write(f"Transcribed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== LYRICS BY VERSE ===\n\n")
            for i, segment in enumerate(segments, 1):
                f.write(f"Verse {i}:\n")
                f.write(f"Time: [{segment['start']:.2f} → {segment['end']:.2f}]\n")
                f.write(f"Lyrics: {segment['text'].strip()}\n\n")
            
            f.write("=== WORD-BY-WORD TIMING ===\n\n")
            for i, line in enumerate(lines, 1):
                f.write(f"Line {i}:\n")
                f.write(f"Time: [{line[0]['start']:.2f} → {line[-1]['end']:.2f}]\n")
                for word in line:
                    f.write(f"  {word['word'].strip():20} [{word['start']:.2f} → {word['end']:.2f}]\n")
                f.write("\n")

        console.print(f"\n[bold green]Lyrics exported to: [/bold green][cyan]{output_file}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Error occurred:[/bold red] {str(e)}")
        console.print("[yellow]Try checking the audio file quality or using a different model.[/yellow]")