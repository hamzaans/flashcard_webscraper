import requests
from bs4 import BeautifulSoup
from firecrawl import FirecrawlApp
from typing import Dict, Any, Optional
import logging
import re
import os
from urllib.parse import urljoin, urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        """Initialize the web scraper with Firecrawl."""
        # Initialize Firecrawl
        firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
        if not firecrawl_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required")
        
        self.firecrawl = FirecrawlApp(api_key=firecrawl_key)
        
        # Fallback session for basic requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str, include_links: bool = False) -> Dict[str, Any]:
        """
        Scrape a webpage using Firecrawl for better content extraction.
        
        Args:
            url: The URL to scrape
            include_links: Whether to include links in the scraped content
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            logger.info(f"Scraping URL with Firecrawl: {url}")
            
            # Use Firecrawl to scrape the URL
            scrape_result = self.firecrawl.scrape_url(
                url,
                params={
                    'formats': ['markdown', 'html'],
                    'includeTags': ['main', 'article', 'section', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
                    'excludeTags': ['nav', 'footer', 'header', 'aside', 'script', 'style'],
                    'onlyMainContent': True,
                    'waitFor': 2000,  # Wait 2 seconds for dynamic content
                }
            )
            
            # Check if Firecrawl response is valid
            if not scrape_result:
                logger.warning(f"Firecrawl returned empty response for {url}, falling back to local scraping")
                return self._fallback_scrape(url)
            
            # Extract content from Firecrawl result
            # Firecrawl returns data directly, not wrapped in a 'data' field
            title = scrape_result.get('metadata', {}).get('title', '')
            description = scrape_result.get('metadata', {}).get('description', '')
            markdown_content = scrape_result.get('markdown', '')
            html_content = scrape_result.get('html', '')
            content = scrape_result.get('content', '')
            
            # If no markdown content, try to extract from HTML
            if not markdown_content and html_content:
                markdown_content = self._html_to_markdown_fallback(html_content)
            
            logger.info(f"Successfully scraped {url} with Firecrawl")
            return {
                'success': True,
                'url': url,
                'title': title,
                'description': description,
                'markdown': markdown_content,
                'html': html_content,
                'content': content,  # Add the main content field
                'metadata': {
                    'title': title,
                    'description': description,
                    'url': url,
                    'source': 'firecrawl'
                }
            }
                
        except Exception as e:
            logger.error(f"Exception while scraping {url} with Firecrawl: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.info("Falling back to local scraping...")
            return self._fallback_scrape(url)
    
    def _fallback_scrape(self, url: str) -> Dict[str, Any]:
        """Fallback to local scraping if Firecrawl fails."""
        try:
            logger.info(f"Using fallback scraping for: {url}")
            
            # Fetch the webpage
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '').strip()
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'menu', 'sidebar']):
                element.decompose()
            
            # Remove elements with common navigation classes/IDs
            nav_selectors = [
                '.nav', '.navigation', '.menu', '.sidebar', '.breadcrumb',
                '.pagination', '.footer', '.header', '.topbar', '.navbar',
                '#nav', '#navigation', '#menu', '#sidebar', '#breadcrumb',
                '#pagination', '#footer', '#header', '#topbar', '#navbar'
            ]
            
            for selector in nav_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Convert to markdown-like text
            markdown_content = self._html_to_markdown(main_content)
            
            logger.info(f"Successfully scraped {url} with fallback method")
            return {
                'success': True,
                'url': url,
                'title': title,
                'description': description,
                'markdown': markdown_content,
                'html': str(main_content),
                'metadata': {
                    'title': title,
                    'description': description,
                    'url': url,
                    'source': 'fallback'
                }
            }
                
        except Exception as e:
            logger.error(f"Exception while fallback scraping {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract the main content from the webpage."""
        # Try to find main content areas
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '#main'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content
        
        # If no main content found, use body
        body = soup.find('body')
        if body:
            return body
        
        return soup
    
    def _html_to_markdown(self, element) -> str:
        """Convert HTML element to markdown-like text."""
        if not element:
            return ""
        
        text = element.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                # Remove common URL patterns and navigation text
                if not self._is_navigation_line(line):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_navigation_line(self, line: str) -> bool:
        """Check if a line contains navigation or irrelevant content."""
        # Common navigation patterns
        nav_patterns = [
            r'^[a-z]+&[a-z]+',  # URL parameters
            r'^[a-z]+#[a-z]+',  # URL fragments
            r'^[a-z]+/[a-z]+',  # URL paths
            r'^[a-z]+\.[a-z]+',  # File extensions
            r'^[a-z]+:[a-z]+',  # Protocols
            r'^[a-z]+\([a-z]+\)',  # Function calls
            r'^[a-z]+\[[a-z]+\]',  # Array access
            r'^[a-z]+\{[a-z]+\}',  # Object access
        ]
        
        for pattern in nav_patterns:
            if re.match(pattern, line.lower()):
                return True
        
        # Common navigation text
        nav_text = [
            'userlogin', 'returnto', 'donate', 'sidebar', 'wmf_medium',
            'wmf_campaign', 'special', 'edit', 'history', 'talk',
            'contributions', 'log', 'create', 'account', 'preferences',
            'watchlist', 'beta', 'vector', 'minerva', 'help', 'articles'
        ]
        
        line_lower = line.lower()
        for nav_word in nav_text:
            if nav_word in line_lower and len(line) < 50:  # Short lines with nav words
                return True
        
        return False
    
    def _html_to_markdown_fallback(self, html_content: str) -> str:
        """Convert HTML to markdown-like text as fallback."""
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        return self._html_to_markdown(soup)
    
    def extract_text_content(self, scrape_result: Dict[str, Any]) -> str:
        """
        Extract clean text content from scraped result.
        
        Args:
            scrape_result: Result from scrape_url method
            
        Returns:
            Clean text content
        """
        if not scrape_result.get('success'):
            return ""
        
        # Prefer markdown content as it's cleaner
        markdown_content = scrape_result.get('markdown', '')
        if markdown_content:
            return markdown_content
        
        # Fallback to HTML content
        html_content = scrape_result.get('html', '')
        if html_content:
            # Basic HTML tag removal (you might want to use BeautifulSoup for better parsing)
            import re
            clean_text = re.sub(r'<[^>]+>', '', html_content)
            return clean_text
        
        return ""
