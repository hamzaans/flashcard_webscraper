# AI-Powered Flashcard Generator

This application uses Firecrawl for enhanced web scraping and content analysis, combined with local Ollama LLM to automatically generate high-quality Anki flashcards from web content. It intelligently identifies relevant terms and concepts based on the topic of the webpage.

## Features

- 🔍 **Enhanced Web Scraping**: Uses Firecrawl API for superior content extraction and analysis
- 🤖 **Hybrid AI Analysis**: Combines Firecrawl's content processing with local spaCy + NLTK + Ollama
- 📚 **Smart Term Extraction**: Firecrawl identifies key concepts, Ollama generates educational flashcards
- 🎴 **Anki Integration**: Generates ready-to-import .apkg files for Anki
- 🎨 **Beautiful Cards**: Creates well-formatted flashcards with topic and source information
- 🚀 **Best of Both Worlds**: Firecrawl's power + local LLM privacy

## Prerequisites

- Python 3.7+
- Firecrawl API key (get from https://firecrawl.dev/)
- Ollama installed locally (for LLM processing)

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Set up your Firecrawl API key:
   ```bash
   # Copy the template and add your API key
   cp env_template.txt .env
   # Edit .env and add your FIRECRAWL_API_KEY
   ```

6. Install and set up Ollama (for local LLM):
   ```bash
   # Install Ollama from https://ollama.ai/
   # Then install a model:
   ollama pull llama3.2
   ```

That's it! You'll need a Firecrawl API key, but Ollama runs locally.

## Usage

### Basic Usage
```bash
python main.py https://en.wikipedia.org/wiki/Mitochondrion
```

### Advanced Usage
```bash
# Limit number of flashcards
python main.py https://example.com/article --max-cards 15

# Custom output filename
python main.py https://example.com/article --output my_biology_cards

# Use a different Ollama model
python main.py https://example.com/article --model llama3.1

# Verbose logging
python main.py https://example.com/article --verbose
```

### Command Line Options
- `url`: The webpage URL to generate flashcards from (required)
- `--max-cards`: Maximum number of flashcards to generate (default: 20)
- `--output`: Custom output filename without .apkg extension
- `--model`: Ollama model to use (default: llama3.2)
- `--verbose`: Enable detailed logging

## How It Works

1. **Enhanced Web Scraping**: Firecrawl extracts clean, structured content from the webpage
2. **Topic Identification**: Local NLP analyzes the content to identify the main subject (biology, history, etc.)
3. **Intelligent Term Extraction**: Firecrawl + spaCy identify the most important terms and concepts
4. **Context Finding**: Locates sentences where each term appears
5. **Smart Flashcard Generation**: Ollama creates educational question-answer pairs with proper definitions
6. **Deck Creation**: Generates an Anki-compatible .apkg file

## Example Output

For a biology webpage about mitochondria, the app might generate flashcards like:

**Question**: What is mitochondria?
**Answer**: Mitochondria is an organelle found in eukaryotic cells that serves as the powerhouse of the cell, producing ATP through cellular respiration.

**Question**: What process occurs in mitochondria?
**Answer**: Cellular respiration occurs in mitochondria, where glucose is broken down to produce ATP energy for the cell.

## Importing to Anki

1. Open Anki
2. Go to File → Import
3. Select the generated .apkg file
4. Click Import
5. Your flashcards will be added to a new deck

## Troubleshooting

### Common Issues

1. **Missing spaCy Model**: Run `python -m spacy download en_core_web_sm`
2. **Ollama Not Running**: Start Ollama service and install a model with `ollama pull llama3.2`
3. **Scraping Failures**: Some websites may block automated scraping. Try different URLs
4. **No Terms Found**: The content might not contain enough relevant terms. Try longer articles
5. **Import Errors**: Make sure you're using a recent version of Anki

### Logs
Use the `--verbose` flag to see detailed logs of the process:
```bash
python main.py https://example.com --verbose
```

## File Structure

```
flashcard_from_website_app/
├── main.py                 # Main application
├── web_scraper.py          # Firecrawl web scraping with fallback
├── content_analyzer.py     # Hybrid AI analysis (Firecrawl + spaCy + NLTK + Ollama)
├── flashcard_generator.py  # Anki deck generation
├── config.py              # Configuration with API key validation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
