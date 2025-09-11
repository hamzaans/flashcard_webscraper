import genanki
import random
import hashlib
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlashcardGenerator:
    def __init__(self):
        """Initialize the Anki flashcard generator."""
        # Create a custom model for our flashcards
        self.model = genanki.Model(
            1607392319,  # Random model ID
            'AI Generated Flashcards',
            fields=[
                {'name': 'Question'},
                {'name': 'Answer'},
                {'name': 'Topic'},
                {'name': 'Source'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '''
                    <div style="font-family: Arial; font-size: 18px; text-align: center; padding: 20px;">
                        <h2 style="color: #2c3e50;">{{Question}}</h2>
                        <div style="margin-top: 20px; font-size: 14px; color: #7f8c8d;">
                            Topic: {{Topic}} | Source: {{Source}}
                        </div>
                    </div>
                    ''',
                    'afmt': '''
                    <div style="font-family: Arial; font-size: 18px; text-align: center; padding: 20px;">
                        <h2 style="color: #2c3e50;">{{Question}}</h2>
                        <hr style="border: 1px solid #bdc3c7; margin: 20px 0;">
                        <div style="font-size: 16px; line-height: 1.6; color: #34495e;">
                            {{Answer}}
                        </div>
                        <div style="margin-top: 20px; font-size: 14px; color: #7f8c8d;">
                            Topic: {{Topic}} | Source: {{Source}}
                        </div>
                    </div>
                    ''',
                },
            ],
            css='''
            .card {
                font-family: Arial;
                font-size: 20px;
                text-align: center;
                color: black;
                background-color: white;
            }
            '''
        )
    
    def create_deck(self, title: str, topic: str, source_url: str) -> genanki.Deck:
        """
        Create a new Anki deck.
        
        Args:
            title: Title for the deck
            topic: Topic/subject of the content
            source_url: URL of the source webpage
            
        Returns:
            Anki deck object
        """
        # Generate a unique deck ID based on title and URL
        deck_id = int(hashlib.md5(f"{title}{source_url}".encode()).hexdigest()[:8], 16)
        
        deck = genanki.Deck(
            deck_id,
            title
        )
        
        logger.info(f"Created deck: {title} (ID: {deck_id})")
        return deck
    
    def add_flashcards(self, deck: genanki.Deck, flashcards: List[Dict[str, str]], topic: str, source_url: str) -> None:
        """
        Add flashcards to the deck.
        
        Args:
            deck: The Anki deck to add cards to
            flashcards: List of flashcard dictionaries with 'question' and 'answer' keys
            topic: Topic/subject of the content
            source_url: URL of the source webpage
        """
        for i, card in enumerate(flashcards):
            try:
                note = genanki.Note(
                    model=self.model,
                    fields=[
                        card['question'],
                        card['answer'],
                        topic,
                        source_url
                    ]
                )
                deck.add_note(note)
                logger.info(f"Added flashcard {i+1}: {card['question'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error adding flashcard {i+1}: {str(e)}")
                continue
    
    def save_deck(self, deck: genanki.Deck, filename: str) -> str:
        """
        Save the deck to an .apkg file.
        
        Args:
            deck: The Anki deck to save
            filename: Name of the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        try:
            output_path = f"{filename}.apkg"
            genanki.Package(deck).write_to_file(output_path)
            logger.info(f"Deck saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving deck: {str(e)}")
            raise
    
    def generate_flashcards_from_content(self, 
                                       content: str, 
                                       topic: str, 
                                       source_url: str,
                                       max_cards: int = 20) -> List[Dict[str, str]]:
        """
        Generate flashcards from content (placeholder for integration with ContentAnalyzer).
        
        Args:
            content: The text content to generate flashcards from
            topic: Topic/subject of the content
            source_url: URL of the source webpage
            max_cards: Maximum number of flashcards to generate
            
        Returns:
            List of flashcard dictionaries
        """
        # This method will be used by the main app to coordinate with ContentAnalyzer
        # For now, return empty list - the actual generation happens in the main app
        return []
