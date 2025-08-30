# AI Voice Agent with Weather Skills

##  Project Overview

This full-stack AI voice assistant provides seamless voice-to-voice conversations with intelligent function calling capabilities. Users can speak naturally to get weather information for any location worldwide and fetch the latest news headlines from various categories.

### Key Capabilities
- **Real-time voice interaction** with streaming transcription and TTS
- **Weather reporting** for any global location using Open-Meteo API
- **News updates** from multiple categories (general, tech, business, sports, science)
- **Conversational memory** with full context retention
- **Dynamic API key configuration** through the UI
- **Modern responsive interface** with real-time status indicators

##  Architecture

```
┌─────────────┐    WebSocket    ┌─────────────────┐
│   Browser   │ ◄──────────────► │  Python Server  │
│             │                 │                 │
│ • HTML/CSS  │                 │ • AssemblyAI    │
│ • JavaScript│                 │ • Gemini AI     │
│ • WebRTC    │                 │ • Murf TTS      │
└─────────────┘                 └─────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   External APIs │
                                │ • Open-Meteo    │
                                │ • RSS Feeds     │
                                └─────────────────┘
```

## Features

### Core Voice Pipeline
- **Streaming Speech-to-Text**: Real-time transcription via AssemblyAI Streaming v3
- **Function Calling**: Intelligent tool use with Google Gemini 1.5 Flash
- **Text-to-Speech**: High-quality voice synthesis using Murf AI

### Smart Skills
- **Weather Reports**: Get current conditions for any city worldwide
- **News Headlines**: Fetch latest news from BBC, TechCrunch, and other sources
- **Natural Conversation**: Contextual responses with memory retention

### Modern Interface
- **Dark theme** with glassmorphism effects
- **Real-time status indicators** for recording, processing, and playback
- **API key configuration** panel for easy setup
- **Responsive design** optimized for desktop and mobile

## 📋 Prerequisites

### Required API Keys
1. **AssemblyAI API Key** - Get from [AssemblyAI Console](https://www.assemblyai.com/)
2. **Google AI API Key** - Get from [Google AI Studio](https://aistudio.google.com/)
3. **Murf AI API Key** - Get from [Murf AI Platform](https://murf.ai/)

### System Requirements
- Python 3.8+
- Modern web browser with microphone access
- Internet connection for API calls

##  Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-voice-agent.git
cd ai-voice-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
ASSEMBLYAI_API_KEY=your_assemblyai_key_here
GOOGLE_API_KEY=your_google_ai_key_here
MURF_API_KEY=your_murf_api_key_here
```

### 4. Run the Application
```bash
python server.py
```

### 5. Open the Client
Open `client.html` in your browser or navigate to `http://localhost:8765`

## Configuration

### API Keys Setup
The application supports both:
- **Environment variables** (`.env` file) for default keys
- **UI configuration** allowing users to input their own API keys

Use the settings panel in the interface to configure API keys without restarting the server.

### Voice Settings
Modify these parameters in `server.py`:
- **Sample rate**: 16000 Hz (default)
- **Audio format**: WAV/PCM
- **Voice ID**: Customizable Murf voice selection

## Usage

### Basic Conversation
1. Click the microphone button to start recording
2. Speak your message clearly
3. The AI will process and respond with voice

### Weather Queries
- "What's the weather in Tokyo?"
- "How's the temperature in New York right now?"
- "Give me the weather forecast for London"

### News Requests
- "Get me the latest tech news"
- "What are today's top headlines?"
- "Show me 5 business news stories"

### Available News Categories
- `general` (default)
- `technology`
- `business` 
- `sports`
- `science`

## Project Structure

```
ai-voice-agent/
├── server.py              # Main WebSocket server
├── client.html            # Frontend interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md             # This file
└── assets/               # Static assets (if any)
```

## Technical Details

### Backend Stack
- **WebSocket Server**: Real-time bidirectional communication
- **AssemblyAI Streaming v3**: Live speech transcription with turn detection
- **Google Gemini 1.5 Flash**: Function calling and conversational AI
- **Murf AI WebSocket**: Streaming text-to-speech synthesis

### Frontend Stack  
- **Vanilla JavaScript**: WebRTC for audio capture and WebSocket communication
- **Tailwind CSS**: Modern utility-first styling
- **Responsive Design**: Works on desktop and mobile browsers

### API Integration
- **Open-Meteo API**: Free weather data (no API key required)
- **RSS Feeds**: News aggregation from reliable sources
- **Function Calling**: Automatic tool selection based on user intent

## Troubleshooting

### Common Issues

**Microphone not working**
- Ensure browser has microphone permissions
- Check if microphone is being used by other applications
- Try refreshing the page

**API errors**
- Verify all API keys are correctly configured
- Check internet connection
- Monitor server logs for detailed error messages

**WebSocket connection fails**
- Ensure server is running on port 8765
- Check firewall settings
- Try restarting the server

### Debug Mode
Enable detailed logging by modifying the log level in `server.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Roadmap

### Planned Features
- [ ] Multi-language support
- [ ] Custom voice training
- [ ] Conversation export
- [ ] Advanced news filtering
- [ ] Weather alerts and notifications
- [ ] Plugin architecture for new skills

### Technical Improvements
- [ ] Redis session storage
- [ ] Rate limiting
- [ ] Audio compression
- [ ] Offline mode capabilities

## Performance Notes

- **Latency**: ~500-1500ms total pipeline latency
- **Audio Quality**: 44.1kHz output, 16kHz input
- **Concurrent Sessions**: Supports multiple simultaneous users
- **Memory Usage**: ~50MB per active session

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black server.py
```

##  License

This project is licensed under the MIT License.

## Acknowledgments

- **30 Days of AI Voice Agents Challenge** for the inspiration
- **AssemblyAI** for excellent real-time transcription
- **Google AI** for powerful language models
- **Murf AI** for high-quality text-to-speech
- **Open-Meteo** for free weather data

**Built with ❤️ for the AI Voice Agent community**

