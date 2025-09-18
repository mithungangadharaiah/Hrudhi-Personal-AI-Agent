# Hrudhi: Personal AI Note-Taking Agent

🤖 **Hrudhi** is your intelligent personal AI assistant for note-taking and retrieval, designed to replace OneNote/sticky notes with smart, context-aware search and learning capabilities.

## 🚀 Two Distribution Approaches

### 🎯 Approach 1: Standalone Executable (Recommended)
- **File Size**: ~500MB-1GB
- **Internet**: Not required after download
- **Setup**: Copy and run - that's it!
- **Best for**: Users who want zero hassle installation

### 🎯 Approach 2: Installer with Dependencies  
- **File Size**: ~50MB initial download
- **Internet**: Required for first setup
- **Setup**: Downloads AI models on first run
- **Best for**: Developers and users comfortable with Python

## 🎨 Features
- 🤖 **Animated Robotic Face** - Cute robot companion with emotions and blinking
- 🎨 **Creative Dark Theme** - Modern, eye-friendly dark interface
- ✨ **Smooth Animations** - Mood changes, status updates, and visual effects  
- 📝 **Intuitive Note-Taking** - Clean text areas with syntax highlighting
- 💾 **Local Privacy-First Storage** - Files stored safely on your Desktop
- 🔍 **AI-Powered Semantic Search** - Find notes by meaning, not just keywords
- 🧠 **Learning System** - Gets smarter with every note you save
- ⚡ **Fast Retrieval** - Instant search using sentence-transformers
- 🪟 **Easy Windows Installation** - Single executable or simple installer

## 🛠️ Quick Start

### Option 1: Standalone Executable
1. Download `installer_standalone/Hrudhi.exe`
2. Copy to your desired location (e.g., `C:\Program Files\Hrudhi\`)
3. Double-click `Hrudhi.exe` to run
4. Your notes will be stored in `~/Desktop/HrudhiNotes`

### Option 2: Installer with Dependencies
1. Ensure Python 3.8+ is installed
2. Download the `installer_with_deps` folder
3. Double-click `Start_Hrudhi.bat` or run `python hrudhi_installer.py`
4. Click "Install Dependencies" on first run
5. Once installed, click "Launch Hrudhi"

## 🏗️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Windows OS (with plans for cross-platform support)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Hrudhi-Personal-AI-Agent.git
cd Hrudhi-Personal-AI-Agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Or on Linux/Mac
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Building Distributions
```bash
# Build standalone executable (~500MB-1GB)
python build/build_standalone.py

# Build installer with dependencies (~50MB)
python build/build_installer.py

# Build both approaches
python build.py --approach both
```

## 📖 Usage Guide

### Adding Notes
1. Open Hrudhi
2. Type your note in the text area
3. Add a topic/context (e.g., "meeting notes", "project ideas")
4. Click "Save Note"

### Searching Notes  
1. Enter keywords or topics in the search box
2. Click "Search"
3. View results with similarity scores
4. Double-click any result to read the full note

### Notes Storage
- **Location**: `~/Desktop/HrudhiNotes`
- **Format**: Individual `.txt` files with timestamps
- **Metadata**: Stored in `embeddings.json` for fast search

## 🧠 How Hrudhi Learns

Hrudhi uses **sentence-transformers** to understand the semantic meaning of your notes:

1. **When you save a note**: Creates an AI embedding (numerical representation)
2. **When you search**: Compares search query with all stored embeddings  
3. **Results**: Returns most semantically similar notes, not just keyword matches
4. **Learning**: Each new note improves the context and relevance of future searches

## 🔧 Technical Details

### Core Technologies
- **GUI**: Tkinter (Python standard library)
- **AI Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Search**: Cosine similarity on embeddings
- **Storage**: JSON metadata + individual text files
- **Packaging**: PyInstaller for Windows executables

### File Structure
```
Hrudhi-Personal-AI-Agent/
├── hrudhi/
│   └── hrudhi.py              # Core AI agent application
├── build/
│   ├── build_standalone.py    # PyInstaller build for exe
│   └── build_installer.py     # Lightweight installer build  
├── main.py                    # Application entry point
├── requirements.txt           # Python package requirements
├── requirements-dev.txt       # Development dependencies
├── build.py                   # Master build script
├── .gitignore                # Git ignore patterns
├── LICENSE                   # MIT License
├── README.md                 # Comprehensive documentation
└── PACKAGING.md             # Build and distribution guide
```

## 🛣️ Roadmap

- [ ] **Enhanced UI**: Modern interface with themes
- [ ] **Auto Topic Extraction**: AI-powered topic detection
- [ ] **Summarization**: Generate summaries of multiple notes
- [ ] **Cross-platform**: Linux and macOS support
- [ ] **Integration**: Calendar and email integration
- [ ] **Voice Notes**: Speech-to-text capabilities
- [ ] **Collaboration**: Shared note spaces (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Hrudhi-Personal-AI-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Hrudhi-Personal-AI-Agent/discussions)

## 🙏 Acknowledgments

- **sentence-transformers** for amazing AI embeddings
- **scikit-learn** for similarity calculations  
- **PyInstaller** for Windows packaging
- **Tkinter** for simple, reliable GUI framework

---

**Made with ❤️ for productivity enthusiasts who value privacy and simplicity.**
