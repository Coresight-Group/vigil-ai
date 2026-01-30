# ============================================================================
# GUNICORN CONFIGURATION FOR VIGIL AI
# Production WSGI server configuration
# ============================================================================

import multiprocessing
import os

# ============================================================================
# SERVER SOCKET
# ============================================================================

# Bind to localhost (Nginx handles external connections)
bind = "127.0.0.1:8000"

# Alternative: Unix socket (slightly faster)
# bind = "unix:/var/run/gunicorn/vigil.sock"

# ============================================================================
# WORKER PROCESSES
# ============================================================================

# Number of worker processes
# Rule of thumb: (2 x CPU cores) + 1
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class - use gevent for async support
worker_class = "sync"  # Options: sync, gevent, eventlet, tornado

# For async operations, use:
# worker_class = "gevent"
# worker_connections = 1000

# Threads per worker (for sync worker)
threads = 2

# Maximum requests per worker before restart (prevents memory leaks)
max_requests = 1000
max_requests_jitter = 50  # Random jitter to prevent all workers restarting at once

# ============================================================================
# TIMEOUTS
# ============================================================================

# Worker timeout (seconds)
timeout = 120

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Keep-alive connections timeout
keepalive = 5

# ============================================================================
# PROCESS NAMING
# ============================================================================

proc_name = "vigil_ai"

# ============================================================================
# SERVER MECHANICS
# ============================================================================

# Daemonize the process (run in background)
daemon = False  # Set to True for production without systemd

# PID file location - configurable via environment
pidfile = os.environ.get("GUNICORN_PID_FILE", "/var/run/gunicorn/vigil.pid")

# User and group to run as (security)
# user = "www-data"
# group = "www-data"

# Working directory - configurable via environment
chdir = os.environ.get("VIGIL_BACKEND_DIR", "/var/www/vigil/backend")

# Preload application
# NOTE: Set to False if you have issues with database connections being forked
# With preload_app=True, database connections created at startup are shared across workers
preload_app = os.environ.get("GUNICORN_PRELOAD", "False").lower() == "true"

# ============================================================================
# LOGGING
# ============================================================================

# Log level - configurable via environment
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Access log file - configurable via environment
accesslog = os.environ.get("GUNICORN_ACCESS_LOG", "/var/log/gunicorn/vigil_access.log")

# Error log file - configurable via environment
errorlog = os.environ.get("GUNICORN_ERROR_LOG", "/var/log/gunicorn/vigil_error.log")

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Capture stdout/stderr in error log
capture_output = True

# ============================================================================
# SSL (if terminating SSL at Gunicorn instead of Nginx)
# ============================================================================

# Uncomment if you want Gunicorn to handle SSL directly
# keyfile = "/etc/letsencrypt/live/your-domain.com/privkey.pem"
# certfile = "/etc/letsencrypt/live/your-domain.com/fullchain.pem"

# ============================================================================
# SECURITY
# ============================================================================

# Limit request line size
limit_request_line = 4094

# Limit request fields
limit_request_fields = 100

# Limit request field size
limit_request_field_size = 8190

# ============================================================================
# HOOKS
# ============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    pass

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    pass

def when_ready(server):
    """Called just after the server is started."""
    print("Vigil AI server is ready. Spawning workers...")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker spawned (pid: {worker.pid})")

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    print(f"Worker {worker.pid} interrupted")

def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    print(f"Worker {worker.pid} aborted")

def pre_exec(server):
    """Called just before a new master process is forked."""
    print("Forked child, re-executing...")

def child_exit(server, worker):
    """Called in master process after worker exits."""
    print(f"Worker {worker.pid} exited")

def worker_exit(server, worker):
    """Called in worker process just after exit."""
    pass

def nworkers_changed(server, new_value, old_value):
    """Called when number of workers changes."""
    print(f"Worker count changed from {old_value} to {new_value}")

def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("Vigil AI server shutting down...")

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Pass environment variables to workers
raw_env = [
    "FLASK_ENV=production",
    "FLASK_DEBUG=False",
]

# ============================================================================
# DEVELOPMENT SETTINGS (override for local testing)
# ============================================================================

# For development, you might want:
# bind = "0.0.0.0:5000"
# workers = 1
# reload = True
# loglevel = "debug"
