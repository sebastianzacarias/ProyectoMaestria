import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG", "False") == "True"

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set")
if not UPLOAD_DIR:
    raise ValueError("UPLOAD_DIR environment variable is not set")
if not DEBUG:
    raise ValueError("DEBUG environment variable is not set")