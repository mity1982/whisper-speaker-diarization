# Audio Transcription with Speaker Diarization

A Python script that transcribes audio files using OpenAI's Whisper and identifies different speakers using machine learning clustering techniques.

## Features

- 🎙️ **Accurate transcription** using OpenAI Whisper (local processing, no API required)
- 👥 **Speaker diarization** - automatically identifies different speakers
- ⏰ **Timestamp support** - precise timing for each speech segment
- 🗣️ **Multi-language support** - currently optimized for Russian
- 🔧 **Configurable** - adjustable chunk size, model quality, and number of speakers
- 💾 **No external dependencies** - runs completely offline after initial setup

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- macOS, Linux, or Windows

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd whisper-speaker-diarization
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg:**
   
   **macOS (using Homebrew):**
   ```bash
   brew install ffmpeg
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```
   
   **Windows:**
   Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Usage

### Basic Usage

```bash
python transcribe_audio.py path/to/your/audio.wav
```

### Interactive Mode

```bash
python transcribe_audio.py
# Script will prompt you to enter the audio file path
```

### Configuration

Edit the script to adjust settings:

```python
CHUNK_LENGTH_MIN = 10      # Audio chunk size in minutes
MODEL_NAME = "base"        # Whisper model: tiny, base, small, medium, large
NUM_SPEAKERS = 2           # Expected number of speakers
OUTPUT_FILE = "transcription.txt"  # Output filename
```

## Whisper Models

| Model  | Size | Speed | Accuracy |
|--------|------|-------|----------|
| tiny   | 39MB | Fastest | Basic |
| base   | 142MB | Fast | Good |
| small  | 466MB | Medium | Better |
| medium | 1.5GB | Slow | Very Good |
| large  | 2.9GB | Slowest | Best |

## Output Format

The script generates a timestamped transcription with speaker identification:

```
[00:00 - 00:05] Спикер 1: У нас будет группы 5 человек и по 4.
[00:05 - 00:11] Спикер 2: Мы группы образуем спортивным образом.
[00:11 - 00:14] Спикер 1: Почитаемся на первую, вторую, первую и первую.
```

## How It Works

1. **Audio Loading**: Uses Whisper's built-in audio loading with FFmpeg
2. **Chunking**: Splits long audio into manageable chunks (default 10 minutes)
3. **Transcription**: Each chunk is transcribed using Whisper
4. **Feature Extraction**: Extracts audio features for each speech segment:
   - Signal energy
   - Zero crossing rate
   - Spectral centroid and bandwidth
   - Mel-frequency coefficients
5. **Speaker Clustering**: Uses K-means clustering to group segments by speaker
6. **Output Generation**: Combines transcription with speaker labels and timestamps

## Troubleshooting

### FFmpeg Not Found
```
[ERROR] FFmpeg не найден!
```
**Solution**: Install FFmpeg following the installation instructions above.

### SSL Certificate Errors
The script automatically handles SSL certificate issues during model download.

### Memory Issues
For very long audio files, consider:
- Using a smaller Whisper model (`tiny` or `base`)
- Reducing `CHUNK_LENGTH_MIN`
- Processing shorter segments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [scikit-learn](https://scikit-learn.org/) for machine learning clustering
- FFmpeg for audio processing

## Roadmap

- [ ] GUI interface
- [ ] Real-time processing
- [ ] More sophisticated speaker diarization
- [ ] Support for more languages
- [ ] Docker containerization
- [ ] Web interface
