# Hybrid configuration - Firecrawl for scraping, Ollama for AI processing

# This application uses:
# - Web scraping: Firecrawl API (for better content extraction)
# - Content analysis: Firecrawl + spaCy + NLTK + Ollama
# - Flashcard generation: genanki

def validate_setup():
    """Validate that the setup is ready."""
    try:
        import spacy
        import nltk
        import genanki
        import requests
        from bs4 import BeautifulSoup
        from firecrawl import FirecrawlApp
        import ollama
        return True
    except ImportError as e:
        raise ImportError(f"Missing required library: {e}. Please install with: pip install -r requirements.txt")

def validate_api_keys():
    """Validate that required API keys are set."""
    import os
    
    # Check for Firecrawl API key
    firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
    if not firecrawl_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable is required. Get your key from: https://firecrawl.dev/")
    
    return True
