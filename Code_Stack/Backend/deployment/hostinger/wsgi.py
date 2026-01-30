# ============================================================================
# WSGI ENTRY POINT FOR HOSTINGER VPS
# This file is the entry point for WSGI servers (Gunicorn, uWSGI)
# ============================================================================

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(backend_dir), 'Keys_Security', 'env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Fallback to standard .env location
    load_dotenv()

# Import the Flask application
from app import app as application

# Expose the WSGI callable
app = application

if __name__ == "__main__":
    application.run()
