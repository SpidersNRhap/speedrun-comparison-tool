# 🎮 Speedrun Comparison Tool

A powerful desktop application for comparing speedrun videos side-by-side with precise timing analysis and automatic video generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📖 Usage Guide

### Loading Videos
- Click the **"Load Video"** button in any video panel
- Supported formats: MP4, AVI, MOV, MKV
- Each video gets its own control panel with independent playback

### Marking Segments
1. Navigate to your desired start frame using:
   - **Playback controls**: ▶ ⏸ buttons
   - **Frame stepping**: `<` `>` for single frames, `<<` `>>` for 10 frames
   - **Scrub bar**: Click and drag for quick seeking
2. Click **"Mark Start"** to set the beginning
3. Navigate to the end frame and click **"Mark End"**
4. Use **"Jump Start"/"Jump End"** to quickly return to marked positions

### Comparing Times
- Click **"Calculate Difference"** to analyze all marked videos
- Results show individual segment times and relative differences
- Times are calculated using each video's native frame rate for accuracy

### Generating Comparison Videos

#### Export Settings
- **Output FPS**: 20, 30, or 60 fps
- **Resolution**: Quarter (0.25x), Half (0.5x), or Full (1.0x)
- **Codec**: Auto-detect, H.264, MPEG-4, or Xvid

#### Generation Process
1. Configure your export settings
2. Click **"Generate Comparison Video"**
3. Choose output location and filename

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

