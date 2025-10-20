#!/usr/bin/env python3
"""
Enhanced main entry point for the Zendesk Roleplay application.

This script:
- Sets up the Python path correctly
- Starts both UI and backend together
- Watches for file changes and restarts services as needed
- Provides graceful shutdown handling
"""

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    # Create dummy classes when watchdog is not available
    class FileSystemEventHandler:
        def on_modified(self, event):
            pass

    class Observer:
        def schedule(self, handler, path, recursive=False):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    WATCHDOG_AVAILABLE = False


def run_backend():
    """Run the FastAPI backend server."""
    try:
        import uvicorn

        from src.core.app import app

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
    except Exception as e:
        print(f"❌ Backend error: {e}")


def run_ui():
    """Run the Gradio UI server."""
    try:
        from src.core.ui import launch_ui

        launch_ui()
    except Exception as e:
        print(f"❌ UI error: {e}")


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events to trigger restarts."""

    def __init__(self, restart_callback):
        super().__init__()
        self.restart_callback = restart_callback
        self.last_restart = 0
        self.debounce_time = 2  # seconds

    def on_modified(self, event):
        if event.is_directory:
            return

        # Only restart for Python files
        if not event.src_path.endswith(".py"):
            return

        # Debounce rapid changes
        current_time = time.time()
        if current_time - self.last_restart < self.debounce_time:
            return

        self.last_restart = current_time
        print(f"🔄 File changed: {event.src_path}")
        self.restart_callback()


class ServiceManager:
    """Manages backend and UI services with restart capabilities using subprocess."""

    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.ui_process: Optional[subprocess.Popen] = None
        self.observer: Optional[Observer] = None
        self.shutdown_flag = False
        self.backend_restart_count = 0
        self.ui_restart_count = 0
        self.max_restarts = 5
        self.restart_delay = 5  # seconds

    def log_process_output(self, process, name):
        """Log output from a subprocess."""
        if process and process.stdout:
            try:
                for line in process.stdout:
                    if line.strip():
                        print(f"[{name}] {line.strip()}")
            except:
                pass

    def start_backend(self):
        """Start the FastAPI backend server using subprocess."""
        if self.backend_restart_count >= self.max_restarts:
            print(f"❌ Backend exceeded max restarts ({self.max_restarts}), giving up")
            return

        try:
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, r'{src_dir}')

try:
    from src.core.app import app
    import uvicorn
    print("🚀 Backend starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
except Exception as e:
    print(f"❌ Backend startup error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""",
            ]

            self.backend_process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Start a thread to log output
            threading.Thread(
                target=self.log_process_output,
                args=(self.backend_process, "Backend"),
                daemon=True,
            ).start()

            print("🚀 Backend process started")
            self.backend_restart_count += 1

        except Exception as e:
            print(f"❌ Failed to start backend: {e}")

    def start_ui(self):
        """Start the Gradio UI server using subprocess."""
        if self.ui_restart_count >= self.max_restarts:
            print(f"❌ UI exceeded max restarts ({self.max_restarts}), giving up")
            return

        try:
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, r'{src_dir}')

try:
    from src.core.ui import launch_ui
    print("🎨 UI starting...")
    launch_ui()
except Exception as e:
    print(f"❌ UI startup error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""",
            ]

            self.ui_process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Start a thread to log output
            threading.Thread(
                target=self.log_process_output,
                args=(self.ui_process, "UI"),
                daemon=True,
            ).start()

            print("🎨 UI process started")
            self.ui_restart_count += 1

        except Exception as e:
            print(f"❌ Failed to start UI: {e}")

    def stop_services(self):
        """Stop all running services."""
        print("🛑 Stopping services...")

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                print("✅ Backend killed")
            except Exception as e:
                print(f"⚠️ Error stopping backend: {e}")
            finally:
                self.backend_process = None

        if self.ui_process:
            try:
                self.ui_process.terminate()
                self.ui_process.wait(timeout=5)
                print("✅ UI stopped")
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
                print("✅ UI killed")
            except Exception as e:
                print(f"⚠️ Error stopping UI: {e}")
            finally:
                self.ui_process = None

        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("✅ File watcher stopped")

    def restart_services(self):
        """Restart all services after file changes."""
        if self.shutdown_flag:
            return

        print("♻️ Restarting services due to file changes...")
        # Stop services but don't join the observer thread since we're in it
        self.stop_services_for_restart()

        # Reset restart counts for file-triggered restarts
        self.backend_restart_count = 0
        self.ui_restart_count = 0

        time.sleep(2)  # Brief pause
        self.start_services()

    def stop_services_for_restart(self):
        """Stop backend and UI services only (not observer) for restart."""
        print("🛑 Stopping services...")

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                print("✅ Backend killed")
            except Exception as e:
                print(f"⚠️ Error stopping backend: {e}")
            finally:
                self.backend_process = None

        if self.ui_process:
            try:
                self.ui_process.terminate()
                self.ui_process.wait(timeout=5)
                print("✅ UI stopped")
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
                print("✅ UI killed")
            except Exception as e:
                print(f"⚠️ Error stopping UI: {e}")
            finally:
                self.ui_process = None

    def start_services(self):
        """Start both backend and UI services."""
        self.start_backend()
        time.sleep(3)  # Let backend start first
        self.start_ui()

    def start_file_watcher(self):
        """Start watching for file changes."""
        if not WATCHDOG_AVAILABLE:
            print("📁 File watching disabled (watchdog not available)")
            return

        handler = FileChangeHandler(self.restart_services)
        self.observer = Observer()

        # Watch src directory for Python file changes
        watch_path = Path(__file__).parent / "src"
        if watch_path.exists():
            self.observer.schedule(handler, str(watch_path), recursive=True)
            self.observer.start()
            print(f"👀 Watching {watch_path} for changes...")

    def run(self, watch_files=True):
        """Main run method to start all services."""

        def signal_handler(signum, frame):
            print(f"\n🔶 Received signal {signum}")
            self.shutdown_flag = True
            self.stop_services()
            sys.exit(0)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            print("🎬 Starting Zendesk Roleplay Application...")
            print("=" * 50)

            # Start services
            self.start_services()

            # Start file watcher if requested
            if watch_files:
                self.start_file_watcher()

            print("=" * 50)
            print("✅ Services started!")
            print("📊 Backend API: http://localhost:8000")
            print("🎨 UI: Check output above for Gradio URL")
            print("⌨️ Press Ctrl+C to stop all services")
            print("=" * 50)

            # Keep main thread alive and monitor processes
            last_check = time.time()
            try:
                while not self.shutdown_flag:
                    time.sleep(2)

                    # Only check process health every 10 seconds to avoid spam
                    if time.time() - last_check > 10:
                        last_check = time.time()

                        # Check if processes are still alive
                        if (
                            self.backend_process
                            and self.backend_process.poll() is not None
                        ):
                            if self.backend_restart_count < self.max_restarts:
                                print(
                                    f"⚠️ Backend process died (exit code: {self.backend_process.poll()})"
                                )
                                print(
                                    f"   Waiting {self.restart_delay}s before restart..."
                                )
                                time.sleep(self.restart_delay)
                                self.start_backend()
                            else:
                                print(
                                    "❌ Backend failed too many times, not restarting"
                                )

                        if self.ui_process and self.ui_process.poll() is not None:
                            if self.ui_restart_count < self.max_restarts:
                                print(
                                    f"⚠️ UI process died (exit code: {self.ui_process.poll()})"
                                )
                                print(
                                    f"   Waiting {self.restart_delay}s before restart..."
                                )
                                time.sleep(self.restart_delay)
                                self.start_ui()
                            else:
                                print("❌ UI failed too many times, not restarting")

            except KeyboardInterrupt:
                pass

        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop_services()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Zendesk Roleplay Application")
    parser.add_argument("--no-watch", action="store_true", help="Disable file watching")
    parser.add_argument("--backend-only", action="store_true", help="Run backend only")
    parser.add_argument("--ui-only", action="store_true", help="Run UI only")

    args = parser.parse_args()

    if args.backend_only:
        print("🚀 Starting backend only...")
        try:
            run_backend()
        except KeyboardInterrupt:
            print("\n👋 Backend stopped")
        return

    if args.ui_only:
        print("🎨 Starting UI only...")
        try:
            run_ui()
        except KeyboardInterrupt:
            print("\n👋 UI stopped")
        return

    # Run both services with optional file watching
    manager = ServiceManager()
    manager.run(watch_files=not args.no_watch)


if __name__ == "__main__":
    main()
