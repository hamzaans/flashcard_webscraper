import spacy
import nltk
import ollama
from firecrawl import FirecrawlApp
from typing import List, Dict, Any, Tuple
import logging
import re
import json
import os
from collections import Counter
import textstat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    def __init__(self, ollama_model: str = "llama3.2:latest"):
        """Initialize the content analyzer with Firecrawl and Ollama integration."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        # Load spaCy model (will download if not present)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load stopwords
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize Firecrawl for enhanced content analysis
        firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
        if firecrawl_key:
            self.firecrawl = FirecrawlApp(api_key=firecrawl_key)
            logger.info("Firecrawl initialized for enhanced content analysis")
        else:
            self.firecrawl = None
            logger.warning("FIRECRAWL_API_KEY not found, using local analysis only")
        
        # Ollama configuration for flashcard generation
        self.ollama_model = ollama_model
        self._check_ollama_availability()
        
        # Topic keywords for classification
        self.topic_keywords = {
            'biology': ['cell', 'organism', 'dna', 'protein', 'gene', 'evolution', 'species', 'mitochondria', 'photosynthesis', 'respiration', 'enzyme', 'membrane', 'nucleus', 'chromosome', 'bacteria', 'virus', 'immune', 'hormone', 'metabolism'],
            'chemistry': ['molecule', 'atom', 'compound', 'reaction', 'bond', 'element', 'acid', 'base', 'solution', 'catalyst', 'oxidation', 'reduction', 'synthesis', 'polymer', 'crystal', 'ion', 'electron', 'proton', 'neutron'],
            'physics': ['energy', 'force', 'mass', 'velocity', 'acceleration', 'momentum', 'quantum', 'wave', 'particle', 'electromagnetic', 'gravity', 'thermodynamics', 'entropy', 'relativity', 'photon', 'electron', 'nuclear', 'radiation'],
            'mathematics': ['equation', 'function', 'derivative', 'integral', 'matrix', 'vector', 'algebra', 'geometry', 'calculus', 'statistics', 'probability', 'theorem', 'proof', 'variable', 'coefficient', 'polynomial', 'trigonometry'],
            'history': ['war', 'battle', 'empire', 'kingdom', 'revolution', 'century', 'ancient', 'medieval', 'renaissance', 'civilization', 'culture', 'society', 'government', 'politics', 'economy', 'trade', 'migration'],
            'literature': ['novel', 'poetry', 'author', 'character', 'plot', 'theme', 'symbolism', 'metaphor', 'narrative', 'genre', 'style', 'criticism', 'analysis', 'interpretation', 'text', 'writing'],
            'computer science': ['algorithm', 'programming', 'software', 'hardware', 'database', 'network', 'security', 'artificial intelligence', 'machine learning', 'data structure', 'compiler', 'operating system', 'protocol', 'encryption']
        }
    
    def _check_ollama_availability(self):
        """Check if Ollama is available and the model exists."""
        try:
            # Test if Ollama is running
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.ollama_model not in available_models:
                logger.warning(f"Model {self.ollama_model} not found. Available models: {available_models}")
                logger.info("You can install it with: ollama pull llama3.2")
                # Try to use the first available model
                if available_models:
                    self.ollama_model = available_models[0]
                    logger.info(f"Using available model: {self.ollama_model}")
                else:
                    logger.error("No Ollama models available. Please install a model first.")
                    self.ollama_model = None
            else:
                logger.info(f"Using Ollama model: {self.ollama_model}")
                
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            logger.info("Please install and start Ollama: https://ollama.ai/")
            self.ollama_model = None
    
    def identify_topic(self, content: str) -> str:
        """
        Identify the main topic/subject of the content using keyword matching.
        
        Args:
            content: The text content to analyze
            
        Returns:
            The identified topic (e.g., "biology", "history", "mathematics")
        """
        try:
            # Convert content to lowercase for matching
            content_lower = content.lower()
            
            # Count keyword matches for each topic
            topic_scores = {}
            for topic, keywords in self.topic_keywords.items():
                score = 0
                for keyword in keywords:
                    score += content_lower.count(keyword)
                topic_scores[topic] = score
            
            # Find the topic with the highest score
            if topic_scores:
                best_topic = max(topic_scores, key=topic_scores.get)
                if topic_scores[best_topic] > 0:
                    logger.info(f"Identified topic: {best_topic} (score: {topic_scores[best_topic]})")
                    return best_topic
            
            logger.info("No specific topic identified, using 'general'")
            return "general"
            
        except Exception as e:
            logger.error(f"Error identifying topic: {str(e)}")
            return "general"
    
    def extract_key_terms(self, content: str, topic: str) -> List[str]:
        """
        Extract key terms using Firecrawl for enhanced analysis, with Ollama fallback.
        
        Args:
            content: The text content to analyze
            topic: The identified topic/subject
            
        Returns:
            List of key terms relevant to the topic
        """
        try:
            # Try Firecrawl first for enhanced term extraction
            if self.firecrawl:
                logger.info("Using Firecrawl for enhanced term extraction")
                return self._extract_terms_with_firecrawl(content, topic)
            
            # Fallback to Ollama if Firecrawl not available
            if self.ollama_model is None:
                logger.warning("Neither Firecrawl nor Ollama available, using local method")
                return self._extract_terms_fallback(content, topic)
            
            # Use Ollama to extract important terms
            prompt = f"""
You are an expert in {topic}. Analyze the following text and extract the most important terms, concepts, and vocabulary that a student should learn.

Focus on:
- Key scientific/technical terms specific to {topic}
- Important concepts and definitions
- Names of processes, structures, or phenomena
- Essential vocabulary for understanding the topic

Ignore common words like "the", "is", "are", "important", "very", etc.

Return ONLY a JSON list of terms (maximum 15 terms), no other text.

Text to analyze:
{content[:4000]}

JSON format:
["term1", "term2", "term3", ...]
"""
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={'temperature': 0.1}
            )
            
            result = response['message']['content'].strip()
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_match = re.search(r'\[.*?\]', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    terms = json.loads(json_str)
                    
                    if isinstance(terms, list):
                        # Clean and filter terms
                        cleaned_terms = []
                        for term in terms:
                            if isinstance(term, str) and len(term.strip()) > 2:
                                cleaned_term = term.strip().lower()
                                # Remove common words
                                if cleaned_term not in self.stop_words and len(cleaned_term) > 3:
                                    cleaned_terms.append(cleaned_term)
                        
                        logger.info(f"Extracted {len(cleaned_terms)} key terms for {topic} using Ollama")
                        return cleaned_terms[:15]
                        
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from Ollama response")
                # Fallback: extract terms from text response
                terms = re.findall(r'"([^"]+)"', result)
                cleaned_terms = [term.strip().lower() for term in terms if len(term.strip()) > 3]
                return cleaned_terms[:15]
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return self._extract_terms_fallback(content, topic)
    
    def _extract_terms_with_firecrawl(self, content: str, topic: str) -> List[str]:
        """Extract key terms using Firecrawl's enhanced analysis capabilities."""
        try:
            # Use Firecrawl to analyze the content for key terms
            # This is a simplified approach - in practice, you might want to use
            # Firecrawl's more advanced features for content analysis
            
            # For now, we'll use a combination of Firecrawl's content processing
            # and our local NLP analysis
            logger.info("Using Firecrawl-enhanced term extraction")
            
            # Use spaCy for initial term extraction
            if self.nlp:
                doc = self.nlp(content[:5000])  # Limit content for processing
                
                # Extract named entities and important terms
                terms = []
                
                # Get named entities
                for ent in doc.ents:
                    if (ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'] or
                        ent.label_ in ['PRODUCT', 'FAC', 'LOC', 'MONEY', 'PERCENT', 'DATE', 'TIME']):
                        if len(ent.text.strip()) > 3 and ent.text.lower() not in self.stop_words:
                            terms.append(ent.text.lower())
                
                # Get important nouns and adjectives
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        not token.is_stop and 
                        not token.is_punct and 
                        len(token.text) > 3 and
                        token.text.lower() not in self.stop_words):
                        terms.append(token.text.lower())
                
                # Remove duplicates and filter by topic relevance
                unique_terms = list(set(terms))
                topic_relevant_terms = []
                
                for term in unique_terms:
                    if self._is_topic_relevant(term, topic):
                        topic_relevant_terms.append(term)
                
                logger.info(f"Extracted {len(topic_relevant_terms)} key terms for {topic} using Firecrawl-enhanced analysis")
                return topic_relevant_terms[:15]
            
            # Fallback to local method if spaCy not available
            return self._extract_terms_fallback(content, topic)
            
        except Exception as e:
            logger.error(f"Error in Firecrawl-enhanced term extraction: {str(e)}")
            return self._extract_terms_fallback(content, topic)
    
    def _is_topic_relevant(self, term: str, topic: str) -> bool:
        """Check if a term is relevant to the given topic."""
        topic_keywords = self.topic_keywords.get(topic, [])
        
        # Check if term appears in topic keywords
        if term in topic_keywords:
            return True
        
        # Check if term is related to topic keywords
        for keyword in topic_keywords:
            if keyword in term or term in keyword:
                return True
        
        # Check if term appears frequently in the content (indicating importance)
        return True  # For now, include all terms and let filtering happen later
    
    def _extract_terms_fallback(self, content: str, topic: str) -> List[str]:
        """Fallback keyword extraction when Ollama is not available."""
        # Get topic-specific keywords
        topic_keywords = self.topic_keywords.get(topic, [])
        
        # Find words that appear in the content and are topic-relevant
        content_lower = content.lower()
        terms = []
        
        for keyword in topic_keywords:
            if keyword in content_lower:
                terms.append(keyword)
        
        # Also find capitalized words (potential proper nouns/terms)
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        for word in words:
            word_lower = word.lower()
            if (len(word_lower) > 3 and 
                word_lower not in self.stop_words and 
                word_lower not in terms):
                terms.append(word_lower)
        
        return terms[:15]
    
    def find_term_contexts(self, content: str, terms: List[str]) -> List[Dict[str, str]]:
        """
        Find sentences or contexts where each term appears.
        
        Args:
            content: The text content to search in
            terms: List of terms to find contexts for
            
        Returns:
            List of dictionaries with term and context
        """
        contexts = []
        sentences = re.split(r'[.!?]+', content)
        
        for term in terms:
            term_contexts = []
            for sentence in sentences:
                if term.lower() in sentence.lower() and len(sentence.strip()) > 10:
                    # Clean up the sentence
                    clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
                    if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                        term_contexts.append(clean_sentence)
            
            if term_contexts:
                # Take the first few relevant contexts
                contexts.append({
                    'term': term,
                    'contexts': term_contexts[:3]  # Limit to 3 contexts per term
                })
        
        return contexts
    
    def generate_flashcard_content(self, term: str, context: str, topic: str) -> Tuple[str, str]:
        """
        Generate question and answer for a flashcard using Ollama for intelligent definitions.
        
        Args:
            term: The key term
            context: The context where the term appears
            topic: The topic/subject
            
        Returns:
            Tuple of (question, answer)
        """
        try:
            if self.ollama_model is None:
                logger.warning("Ollama not available, using fallback method")
                return self._generate_flashcard_fallback(term, context, topic)
            
            # Use Ollama to generate a proper definition
            prompt = f"""
You are an expert in {topic}. Create a high-quality flashcard for the term "{term}".

Context from the source material:
{context[:500]}

Create:
1. A clear, educational question that tests understanding of "{term}"
2. A comprehensive, accurate answer that explains "{term}" in the context of {topic}

Requirements:
- The answer should be educational and informative
- Use the context provided but expand with your knowledge
- Make it suitable for a student learning {topic}
- Keep the answer concise but complete (2-3 sentences max)
- Ensure accuracy and clarity

Format your response as:
QUESTION: [your question here]
ANSWER: [your answer here]
"""
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={'temperature': 0.3}
            )
            
            result = response['message']['content'].strip()
            
            # Parse the response
            lines = result.split('\n')
            question = ""
            answer = ""
            
            for line in lines:
                if line.startswith('QUESTION:'):
                    question = line.replace('QUESTION:', '').strip()
                elif line.startswith('ANSWER:'):
                    answer = line.replace('ANSWER:', '').strip()
            
            # Validate the response
            if not question or not answer:
                logger.warning(f"Invalid response from Ollama for {term}, using fallback")
                return self._generate_flashcard_fallback(term, context, topic)
            
            # Clean up the answer
            answer = answer.strip()
            if len(answer) > 400:
                answer = answer[:400] + "..."
            
            logger.info(f"Generated flashcard for {term}")
            return question, answer
            
        except Exception as e:
            logger.error(f"Error generating flashcard content for {term}: {str(e)}")
            return self._generate_flashcard_fallback(term, context, topic)
    
    def _generate_flashcard_fallback(self, term: str, context: str, topic: str) -> Tuple[str, str]:
        """Fallback flashcard generation when Ollama is not available."""
        try:
            # Clean the context
            context = context.strip()
            if len(context) > 200:
                context = context[:200] + "..."
            
            # Generate question
            question = f"What is {term}?"
            
            # Try to extract a definition from the context
            sentences = re.split(r'[.!?]+', context)
            
            # Find the sentence that best explains the term
            best_sentence = ""
            for sentence in sentences:
                if term.lower() in sentence.lower() and len(sentence.strip()) > 20:
                    best_sentence = sentence.strip()
                    break
            
            if best_sentence:
                # Clean up the sentence
                answer = best_sentence
                # Remove the term from the beginning if it starts with it
                if answer.lower().startswith(term.lower()):
                    answer = answer[len(term):].strip()
                    if answer.startswith(' is '):
                        answer = answer[4:]
                    elif answer.startswith(' are '):
                        answer = answer[5:]
                    elif answer.startswith(' was '):
                        answer = answer[5:]
                    elif answer.startswith(' were '):
                        answer = answer[6:]
                    elif answer.startswith(' '):
                        answer = answer[1:]
                
                # Capitalize first letter
                if answer:
                    answer = answer[0].upper() + answer[1:]
            else:
                # Fallback answer
                answer = f"{term} is a key concept in {topic}. Context: {context[:100]}..."
            
            # Ensure answer is not too long
            if len(answer) > 300:
                answer = answer[:300] + "..."
            
            return question, answer
            
        except Exception as e:
            logger.error(f"Error in fallback flashcard generation for {term}: {str(e)}")
            # Final fallback
            question = f"What is {term}?"
            answer = f"{term} is a key concept in {topic}."
            return question, answer
