#!/usr/bin/env python3
"""
Example script demonstrating how to use the FlashcardApp programmatically.
"""

import os
from main import FlashcardApp

def example_usage():
    """Example of how to use the FlashcardApp class directly."""
    
    # Check for required API key
    if not os.getenv('FIRECRAWL_API_KEY'):
        print("❌ Please set FIRECRAWL_API_KEY environment variable")
        print("   You can create a .env file with your Firecrawl API key")
        print("   Get your key from: https://firecrawl.dev/")
        return
    
    print("🚀 Starting AI flashcard generation with Firecrawl + Ollama...")
    
    # Initialize the app
    app = FlashcardApp()
    
    # Example URLs for different subjects
    example_urls = [
        "https://en.wikipedia.org/wiki/Mitochondrion",  # Biology
        "https://en.wikipedia.org/wiki/Photosynthesis", # Biology
        "https://en.wikipedia.org/wiki/World_War_II",   # History
        "https://en.wikipedia.org/wiki/Quantum_mechanics", # Physics
    ]
    
    print("🎯 AI Flashcard Generator - Example Usage")
    print("=" * 50)
    
    for i, url in enumerate(example_urls, 1):
        print(f"\n📚 Example {i}: Processing {url}")
        print("-" * 30)
        
        try:
            # Process the URL
            output_path = app.process_url(
                url=url,
                max_cards=10,  # Limit to 10 cards for examples
                output_filename=f"example_{i}_cards"
            )
            
            print(f"✅ Success! Generated: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {url}: {str(e)}")
            continue
    
    print("\n🎉 Example completed!")
    print("💡 Check the generated .apkg files and import them into Anki")

if __name__ == "__main__":
    example_usage()
