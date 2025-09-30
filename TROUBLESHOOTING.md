# Troubleshooting Guide

## ✅ Fixed: Import Errors and Infinite Restart Loop

### Problem
The application was experiencing:
- Infinite restart loops with both backend and UI processes dying immediately
- Multiple import errors:
  - `ImportError: cannot import name 'get_db_connection' from 'src.common.storage'`
  - `ImportError: cannot import name 'launch_ui' from 'src.core.ui'`
- Process exit code 1 repeatedly

### Root Causes
**Issue 1:** The `src/common/__init__.py` file was trying to import functions from `storage.py` that didn't exist:
- `get_db_connection` - doesn't exist in storage.py
- `initialize_database` - doesn't exist in storage.py
- `save_chat_session` - doesn't exist in storage.py
- `get_session_messages` - doesn't exist in storage.py
- `clear_session_messages` - doesn't exist in storage.py
- `list_user_sessions` - doesn't exist in storage.py
- `delete_session` - doesn't exist in storage.py

**Issue 2:** The `src/core/ui.py` file was missing the `launch_ui()` function that `__init__.py` was trying to import.

**Issue 3:** Missing `scenario.yml` file in the root directory (app.py was looking for it there).

### Solutions
**Fix 1:** Updated `src/common/__init__.py` to only import functions that actually exist in `storage.py`:

```python
from .storage import (
    add_milestone,
    end_session,
    get_session,
    get_session_milestones,
    get_session_transcript,
    log_turn,
    new_session,
    update_session,
)
```

**Fix 2:** Added `launch_ui()` function to `src/core/ui.py`:

```python
def launch_ui():
    """Launch the Gradio UI server."""
    demo.launch(server_port=7860, show_api=False, share=False)
```

**Fix 3:** Copied `scenario.yml` from `src/core/` to root directory.

### Verification
After the fix:
- ✅ Import errors resolved
- ✅ No more infinite restart loops
- ✅ Process errors now show actual dependency issues (expected)

## Current Status

### Working ✅
- Import structure is correct
- Main.py startup logic works
- Process management works
- Error logging shows actual issues

### Next Steps 📋
To run the application, install the missing dependencies:

```bash
# Install from requirements.txt or individual packages:
pip install fastapi uvicorn loguru sqlalchemy psycopg gradio sentence-transformers torch
```

### Expected Dependencies
Based on the import errors, you need:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `loguru` - Logging
- `sqlalchemy` - Database ORM
- `psycopg` - PostgreSQL adapter
- `gradio` - UI framework
- `sentence-transformers` - ML models
- `torch` - ML framework
- `anthropic` - Claude API client
- `highlight-io` - Application monitoring
- `watchdog` - File watching (optional)
- `structlog` - Structured logging (optional)
- `langgraph` - Graph-based workflows
- `pydantic` - Data validation
- `requests` - HTTP client

## Development Workflow

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run application**: `python main.py`
3. **Individual services**:
   - Backend only: `python main.py --backend-only`
   - UI only: `python main.py --ui-only`
4. **Disable file watching**: `python main.py --no-watch`

## Import Structure Summary

The project now uses proper relative imports:

```
src/
├── common/          # Utilities (circuit_breaker, storage, etc.)
├── core/           # Main logic (app.py, intents.py, etc.)
├── data/           # Data processing
├── db/             # Database utilities
└── test/           # Test modules
```

All modules use relative imports (`.`, `..`) within the src package, making the structure portable and avoiding circular import issues.