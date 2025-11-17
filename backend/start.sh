#!/bin/bash

# Pollen Predictor API - Startup Script

echo "🌸 Starting Pollen Predictor API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if models exist
if [ ! -d "../models" ]; then
    echo "⚠️  Warning: Models directory not found at ../models/"
    echo "    Make sure trained models are available before running predictions."
fi

# Start the server
echo "🚀 Starting FastAPI server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
