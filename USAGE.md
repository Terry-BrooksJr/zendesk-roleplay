# Zendesk Roleplay Application Usage

## Project Structure (Optimized)

The project has been restructured for better separation of concerns:

```
src/
├── common/          # Shared utilities (circuit breaker, caching, validation)
├── core/           # Main business logic (app, providers, intents)
├── data/           # Data models and processing
├── db/             # Database utilities
└── test/           # Test modules

main.py             # Enhanced entry point with file watching
```

## Running the Application

### Option 1: Run Both UI and Backend (Recommended)

```bash
# Install dependencies first
pip install watchdog  # Optional, for file watching

# Run both services with file watching
python main.py

# Run without file watching
python main.py --no-watch
```

### Option 2: Run Individual Services

```bash
# Backend only (FastAPI)
python main.py --backend-only

# UI only (Gradio)
python main.py --ui-only
```

### Option 3: Run Individual Modules

```bash
# From project root
python -m src.core.app    # Backend
python -m src.core.ui     # UI

# Or from src directory
cd src
python -m core.app        # Backend
python -m core.ui         # UI
```

## Features

### Enhanced main.py Features:
- ✅ **Dual Service Management**: Starts both FastAPI backend and Gradio UI
- ✅ **File Watching**: Automatically restarts services when Python files change
- ✅ **Process Management**: Graceful shutdown with Ctrl+C
- ✅ **Health Monitoring**: Automatically restarts crashed services
- ✅ **Flexible Options**: Run individual services or both together

### Import Structure Fixes:
- ✅ **Relative Imports**: Proper relative imports within the src package
- ✅ **Module Execution**: Can run individual files or as modules
- ✅ **Package Structure**: All directories are proper Python packages

## Development Workflow

1. **Start Development Environment**:
   ```bash
   python main.py
   ```
   This starts both backend and UI with file watching.

2. **Make Changes**: Edit any Python file in the `src/` directory

3. **Auto-Restart**: Services automatically restart when files change

4. **Access Services**:
   - Backend API: http://localhost:8000
   - UI: Check console for Gradio URL (usually http://localhost:7860)

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --no-watch      Disable automatic file watching and restart
  --backend-only  Run only the FastAPI backend
  --ui-only       Run only the Gradio UI
  --help          Show help message
```

## Import Examples

### From Project Root:
```python
from src.core.app import app
from src.common.circuit_breaker import CircuitBreaker
from src.data.anthropic_model_prompt import ChatClient
```

### Within src Package (relative imports):
```python
# In src/core/app.py
from .intents import detect
from ..common.storage import get_session
from ..data.anthropic_model_prompt import ChatClient
```

## Testing

```bash
# Run all tests
python -m pytest src/test/

# Run specific test
python -m pytest src/test/test_app.py
```

## Troubleshooting

### ImportError: No module named 'src'
- Make sure you're running from the project root directory
- Use the main.py entry point: `python main.py`

### Missing Dependencies
- Install required packages: `pip install -r requirements.txt`
- Install optional file watching: `pip install watchdog`

### Port Already in Use
- Backend uses port 8000, UI uses port 7860
- Kill existing processes: `lsof -ti:8000 | xargs kill -9`