#!/bin/bash

# AI Flashcard Generator - Run Script
# This script activates the virtual environment and runs the application

echo "🎯 AI Flashcard Generator"
echo "========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# No .env file needed for local-only version!
echo "🚀 Running in local-only mode - no API keys required!"

# Activate virtual environment and run the app
echo "🚀 Starting application..."
source venv/bin/activate

# Pass all arguments to the main script
python main.py "$@"
