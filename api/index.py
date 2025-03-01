from flask import Flask
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from app.py
from app import app

# This is the handler that Vercel will use
# Do not modify this function name or signature
def handler(request):
    """
    Vercel serverless function handler that exposes the Flask app
    """
    return app 