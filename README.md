# 🎵 Lyrics Transcriber

A powerful Python tool for transcribing song lyrics with precise timestamps using OpenAI's Whisper model. Get detailed word-by-word timing and verse analysis of your favorite songs! Perfect for musicians, content creators, and music enthusiasts.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

## ✨ Key Features

- 🎯 High-precision word-by-word timestamp analysis
- 📝 Smart verse detection and transcription
- 🎮 User-friendly interactive file selection
- 🚀 Blazing fast GPU acceleration support
- 📊 Advanced confidence scoring system
- 💾 Robust auto-save and recovery
- 📤 Clean, formatted text exports
- 🎨 Modern, intuitive console interface
- 🔄 Batch processing support
- 🌐 Multi-language support

## 🚀 Quick Start

1. Clone the repository:

```bash
git clone https://github.com/NojinNojs/lyrics-transcriber.git
cd lyrics-transcriber
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages:
  - whisper
  - torch
  - rich
  - pydub
  - tqdm

## 💻 Usage

1. Place your audio files in the project directory
2. Run the script:

```bash
python main.py
```

3. Select your audio file from the interactive menu
4. Wait for the transcription to complete
5. Results will be displayed and saved to a text file

## 📂 Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)

## 📤 Output Format

The script generates two types of analysis:

### 1. Verse Analysis

```bash
Verse 1:
Time: [00:00 → 00:05] (5.00s)
Lyrics: Example lyrics here
Confidence: 95%
```

### 2. Word-by-Word Analysis

```bash
Line 1:
Time: [00:00 → 00:05] (5.00s)
Word: "Example" [00:00 → 00:02]
Word: "lyrics" [00:02 → 00:03]
Word: "here" [00:03 → 00:05]
```


## 🛠️ Technical Details

- Uses OpenAI's Whisper model for transcription
- Supports GPU acceleration via CUDA
- Implements smart download resumption
- Configurable confidence thresholds
- Automatic audio format detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for 
details.

## 👨‍💻 Author

**Nojs Nojin**
- GitHub: [@NojinNojs](https://github.com/NojinNojs)
- Website: [nojin.site](https://nojin.site)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues 
page](https://github.com/NojinNojs/lyrics-transcriber/issues).

## 📜 Changelog

### Version 1.0.0
- Initial release
- Basic transcription functionality
- GPU support
- File selection menu
- Export capability

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Rich library for beautiful console output
- PyDub for audio processing

---
Made with ❤️ by [Nojs Nojin](https://github.com/NojinNojs)