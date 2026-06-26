"""
run_no_debug.py

Convenience runner that starts the Flask app WITHOUT the debug
auto-reloader. Useful in containerized or sandboxed environments where
the file-watcher (inotify) can be noisy or restricted.

Usage: python run_no_debug.py
"""

from app import app

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
