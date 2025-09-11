#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import genanki
        print("✅ genanki imported successfully")
    except ImportError as e:
        print(f"❌ genanki import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4 imported successfully")
    except ImportError as e:
        print(f"❌ beautifulsoup4 import failed: {e}")
        return False
    
    try:
        import spacy
        print("✅ spacy imported successfully")
    except ImportError as e:
        print(f"❌ spacy import failed: {e}")
        return False
    
    try:
        import nltk
        print("✅ nltk imported successfully")
    except ImportError as e:
        print(f"❌ nltk import failed: {e}")
        return False
    
    try:
        import ollama
        print("✅ ollama imported successfully")
    except ImportError as e:
        print(f"❌ ollama import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import validate_setup
        print("✅ Configuration module loaded successfully")
        
        # Test local setup validation
        validate_setup()
        print("✅ Local setup validation passed")
        
        # Test Ollama availability
        try:
            import ollama
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            if available_models:
                print(f"✅ Ollama is running with models: {', '.join(available_models[:3])}")
            else:
                print("⚠️  Ollama is running but no models installed")
                print("   Install a model with: ollama pull llama3.2")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Install Ollama from: https://ollama.ai/")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_custom_modules():
    """Test if our custom modules can be imported."""
    print("\n📦 Testing custom modules...")
    
    try:
        from web_scraper import WebScraper
        print("✅ WebScraper imported successfully")
    except ImportError as e:
        print(f"❌ WebScraper import failed: {e}")
        return False
    
    try:
        from content_analyzer import ContentAnalyzer
        print("✅ ContentAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ ContentAnalyzer import failed: {e}")
        return False
    
    try:
        from flashcard_generator import FlashcardGenerator
        print("✅ FlashcardGenerator imported successfully")
    except ImportError as e:
        print(f"❌ FlashcardGenerator import failed: {e}")
        return False
    
    try:
        from main import FlashcardApp
        print("✅ FlashcardApp imported successfully")
    except ImportError as e:
        print(f"❌ FlashcardApp import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 AI Flashcard Generator - Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test configuration
    if not test_config():
        all_passed = False
    
    # Test custom modules
    if not test_custom_modules():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The application is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Set up your Firecrawl API key in .env file")
        print("   2. Run: python main.py <URL>")
        print("   3. Import the generated .apkg file into Anki")
        print("\n🚀 Firecrawl + Ollama hybrid approach for best results!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Install spaCy model: python -m spacy download en_core_web_sm")
        print("   3. Check Python version (3.7+ required)")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
