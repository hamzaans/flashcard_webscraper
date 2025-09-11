#!/usr/bin/env python3
"""
AI-Powered Flashcard Generator from Web Content

This application uses Firecrawl for enhanced web scraping and content analysis,
combined with local Ollama LLM to identify topics, extract relevant terms,
and generate high-quality Anki flashcards using genanki.
"""

import sys
import os
import argparse
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from config import validate_setup, validate_api_keys
from web_scraper import WebScraper
from content_analyzer import ContentAnalyzer
from flashcard_generator import FlashcardGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlashcardApp:
    def __init__(self, ollama_model: str = "llama3.2"):
        """Initialize the flashcard generation application."""
        self.web_scraper = WebScraper()
        self.content_analyzer = ContentAnalyzer(ollama_model=ollama_model)
        self.flashcard_generator = FlashcardGenerator()
    
    def process_url(self, url: str, max_cards: int = 20, output_filename: str = None) -> str:
        """
        Process a URL and generate flashcards.
        
        Args:
            url: The URL to process
            max_cards: Maximum number of flashcards to generate
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to the generated .apkg file
        """
        logger.info(f"Starting flashcard generation for: {url}")
        
        # Step 1: Scrape the webpage
        logger.info("Step 1: Scraping webpage...")
        scrape_result = self.web_scraper.scrape_url(url)
        
        if not scrape_result['success']:
            raise Exception(f"Failed to scrape URL: {scrape_result.get('error', 'Unknown error')}")
        
        # Extract text content
        content = self.web_scraper.extract_text_content(scrape_result)
        if not content:
            raise Exception("No content extracted from the webpage")
        
        logger.info(f"Extracted {len(content)} characters of content")
        
        # Step 2: Analyze content and identify topic
        logger.info("Step 2: Analyzing content...")
        topic = self.content_analyzer.identify_topic(content)
        logger.info(f"Identified topic: {topic}")
        
        # Step 3: Extract key terms
        logger.info("Step 3: Extracting key terms...")
        key_terms = self.content_analyzer.extract_key_terms(content, topic)
        logger.info(f"Found {len(key_terms)} key terms: {key_terms[:5]}...")
        
        if not key_terms:
            raise Exception("No key terms found in the content")
        
        # Step 4: Find contexts for terms
        logger.info("Step 4: Finding term contexts...")
        term_contexts = self.content_analyzer.find_term_contexts(content, key_terms)
        logger.info(f"Found contexts for {len(term_contexts)} terms")
        
        # Step 5: Generate flashcards
        logger.info("Step 5: Generating flashcards...")
        flashcards = []
        
        for term_data in term_contexts[:max_cards]:  # Limit to max_cards
            term = term_data['term']
            contexts = term_data['contexts']
            
            # Use the first context for flashcard generation
            if contexts:
                context = contexts[0]
                question, answer = self.content_analyzer.generate_flashcard_content(term, context, topic)
                
                flashcards.append({
                    'question': question,
                    'answer': answer,
                    'term': term
                })
        
        logger.info(f"Generated {len(flashcards)} flashcards")
        
        # Step 6: Create Anki deck
        logger.info("Step 6: Creating Anki deck...")
        
        # Generate deck title and filename
        title = scrape_result.get('title', 'AI Generated Flashcards')
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"flashcards_{topic}_{timestamp}"
        
        deck = self.flashcard_generator.create_deck(title, topic, url)
        self.flashcard_generator.add_flashcards(deck, flashcards, topic, url)
        
        # Step 7: Save deck
        logger.info("Step 7: Saving Anki deck...")
        output_path = self.flashcard_generator.save_deck(deck, output_filename)
        
        logger.info(f"✅ Successfully generated {len(flashcards)} flashcards!")
        logger.info(f"📁 Deck saved to: {output_path}")
        logger.info(f"📚 Topic: {topic}")
        logger.info(f"🔗 Source: {url}")
        
        return output_path
    
    def print_summary(self, url: str, output_path: str, topic: str, num_cards: int):
        """Print a summary of the generated flashcards."""
        print("\n" + "="*60)
        print("🎉 FLASHCARD GENERATION COMPLETE!")
        print("="*60)
        print(f"📚 Topic: {topic}")
        print(f"🔗 Source URL: {url}")
        print(f"📝 Number of cards: {num_cards}")
        print(f"📁 Output file: {output_path}")
        print("\n💡 To use these flashcards:")
        print("   1. Open Anki")
        print("   2. Go to File > Import")
        print(f"   3. Select: {output_path}")
        print("   4. Click Import")
        print("="*60)

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from web content using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://en.wikipedia.org/wiki/Mitochondrion
  python main.py https://example.com/article --max-cards 15
  python main.py https://example.com/article --output my_cards
        """
    )
    
    parser.add_argument(
        'url',
        help='URL of the webpage to generate flashcards from'
    )
    
    parser.add_argument(
        '--max-cards',
        type=int,
        default=20,
        help='Maximum number of flashcards to generate (default: 20)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (without .apkg extension)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.2:latest',
        help='Ollama model to use (default: llama3.2:latest)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate setup and API keys
        validate_setup()
        validate_api_keys()
        
        # Initialize the app
        app = FlashcardApp(ollama_model=args.model)
        
        # Process the URL
        output_path = app.process_url(
            url=args.url,
            max_cards=args.max_cards,
            output_filename=args.output
        )
        
        # Extract info for summary
        topic = "Unknown"
        num_cards = 0
        
        # Try to get topic and card count from the generated deck
        try:
            import genanki
            deck = genanki.Deck.from_collection_file(output_path)
            # This is a simplified approach - in practice, you'd need to parse the deck
        except:
            pass
        
        # Print summary
        app.print_summary(args.url, output_path, topic, num_cards)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
