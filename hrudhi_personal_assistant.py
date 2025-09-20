"""
Hrudhi Personal Assistant - Advanced AI-Powered Note Taking & Chat
Modern Windows 11 design with glassmorphism, 3D avatar, AI capabilities, and integrated chat
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import customtkinter as ctk
import pygame
import math
import threading
import queue
import time
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import uuid
import re
import requests
from urllib.parse import urlparse
import webbrowser

# AI and ML imports with graceful fallback
AI_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
    print("🤖 AI capabilities loaded successfully")
except ImportError as e:
    print(f"⚠️ AI features not available: {e}")
    print("💡 You can still use the note-taking features without AI chat")
except Exception as e:
    print(f"⚠️ AI loading interrupted: {e}")
    print("💡 You can still use the note-taking features without AI chat")
    from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Note:
    """Enhanced note data structure"""
    id: str
    title: str
    content: str
    summary: str
    created_at: str
    updated_at: str
    tags: List[str]
    color: str = "#6366F1"  # Modern indigo
    pinned: bool = False
    is_markdown: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Note':
        return cls(**data)

class AIEngine:
    """Enhanced AI engine with chat, summarization, and learning capabilities"""
    
    def __init__(self):
        self.chat_model = None
        self.chat_tokenizer = None
        self.chat_pipeline = None
        self.summarizer = None
        self.embedder = None
        self.notes_data = []
        self.note_embeddings = []
        self.learning_data = []  # Store new learning data
        self.link_data = []  # Store data from shared links
        self.conversation_history = []
        self.load_models()
    
    def load_models(self):
        """Load AI models for chat, search, and summarization"""
        global AI_AVAILABLE
        
        if not AI_AVAILABLE:
            logger.info("🤖 AI features disabled - running in note-taking mode only")
            return
            
        try:
            logger.info("🤖 Loading advanced AI models...")
            
            # Load chat model - Try smaller model first, fallback to larger one
            model_options = [
                "gpt2",  # Very small, always available
                "microsoft/DialoGPT-small",  # Small chat model
                "Qwen/Qwen2.5-7B-Instruct"   # Preferred larger model
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Loading chat model: {model_name}")
                    
                    self.chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.chat_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # Use float32 for compatibility
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    logger.info(f"✅ Successfully loaded {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    if model_name == model_options[-1]:  # Last model failed
                        raise e
                    continue
            
            # Setup chat pipeline
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.chat_model,
                tokenizer=self.chat_tokenizer,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=150,
                repetition_penalty=1.1,
                device=-1  # Force CPU for compatibility
            )
            
            # Load embedder for semantic search and learning
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load summarization model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Advanced summarizer failed, using simpler model: {e}")
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1
                )
            
            logger.info("✅ All AI models loaded successfully!")
            self.load_knowledge_base()
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            self.chat_model = None
            self.summarizer = None
            self.embedder = None
            AI_AVAILABLE = False
    
    def load_knowledge_base(self):
        """Load existing notes and learning data"""
        try:
            # Load notes
            notes_dir = Path("notes_data")
            if notes_dir.exists():
                self.notes_data = []
                notes_text = []
                
                for json_file in notes_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            note = json.load(f)
                            self.notes_data.append(note)
                            text = f"{note.get('title', '')} {note.get('content', '')}"
                            notes_text.append(text)
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")
                
                if notes_text and self.embedder:
                    self.note_embeddings = self.embedder.encode(notes_text)
                    logger.info(f"📚 Indexed {len(notes_text)} notes for AI chat")
            
            # Load learning data
            learning_file = Path("learning_data.json")
            if learning_file.exists():
                try:
                    with open(learning_file, 'r', encoding='utf-8') as f:
                        self.learning_data = json.load(f)
                    logger.info(f"🧠 Loaded {len(self.learning_data)} learning entries")
                except Exception as e:
                    logger.warning(f"Error loading learning data: {e}")
            
            # Load link data
            links_file = Path("link_data.json")
            if links_file.exists():
                try:
                    with open(links_file, 'r', encoding='utf-8') as f:
                        self.link_data = json.load(f)
                    logger.info(f"🔗 Loaded {len(self.link_data)} shared links")
                except Exception as e:
                    logger.warning(f"Error loading link data: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    def add_learning_data(self, data: str, source: str = "user"):
        """Add new learning data to the AI's knowledge base"""
        try:
            learning_entry = {
                "content": data,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            self.learning_data.append(learning_entry)
            
            # Re-encode embeddings if we have embedder
            if self.embedder:
                all_text = [entry["content"] for entry in self.learning_data]
                if all_text:
                    self.learning_embeddings = self.embedder.encode(all_text)
            
            # Save to file
            learning_file = Path("learning_data.json")
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🧠 Added new learning data from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding learning data: {e}")
            return False
    
    def add_link_data(self, url: str, title: str = "", content: str = ""):
        """Add shared link data to knowledge base"""
        try:
            # Try to extract content from URL if not provided
            if not content and url:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Simple text extraction (could be enhanced with BeautifulSoup)
                        content = response.text[:2000]  # First 2000 chars
                except Exception as e:
                    logger.warning(f"Could not fetch content from {url}: {e}")
            
            link_entry = {
                "url": url,
                "title": title or urlparse(url).netloc,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            self.link_data.append(link_entry)
            
            # Save to file
            links_file = Path("link_data.json")
            with open(links_file, 'w', encoding='utf-8') as f:
                json.dump(self.link_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🔗 Added link data: {title or url}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding link data: {e}")
            return False
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant content across all knowledge sources"""
        relevant_content = []
        
        try:
            if not self.embedder:
                return relevant_content
            
            query_embedding = self.embedder.encode([query])
            
            # Search notes
            if hasattr(self, 'note_embeddings') and len(self.note_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.note_embeddings)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        note = self.notes_data[idx].copy()
                        note['similarity'] = similarities[idx]
                        note['source_type'] = 'note'
                        relevant_content.append(note)
            
            # Search learning data
            if hasattr(self, 'learning_embeddings') and len(self.learning_embeddings) > 0:
                similarities = cosine_similarity(query_embedding, self.learning_embeddings)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.3:
                        entry = self.learning_data[idx].copy()
                        entry['similarity'] = similarities[idx]
                        entry['source_type'] = 'learning'
                        relevant_content.append(entry)
            
            # Sort by similarity
            relevant_content.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            return relevant_content[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return relevant_content
    
    def generate_chat_response(self, user_message: str) -> str:
        """Generate intelligent chat response using all available knowledge"""
        if not self.chat_pipeline:
            return "🤖 AI chat is not available. AI models are not loaded. You can still use the note-taking features!"
        
        try:
            # Search for relevant content
            relevant_content = self.search_relevant_content(user_message, top_k=3)
            
            # Build context
            context_info = ""
            if relevant_content:
                context_info = "Context from your knowledge base:\n"
                for i, item in enumerate(relevant_content):
                    source_type = item.get('source_type', 'unknown')
                    if source_type == 'note':
                        context_info += f"- {item.get('title', 'Untitled')}: {item.get('content', '')[:150]}...\n"
                    elif source_type == 'learning':
                        context_info += f"- Learning: {item.get('content', '')[:150]}...\n"
                context_info += "\n"
            
            # Create a simple text prompt (works better with GPT-2)
            prompt = "You are Hrudhi 🤖, an AI assistant. "
            if context_info:
                prompt += f"{context_info}Question: {user_message}\nHrudhi: "
            else:
                prompt += f"User: {user_message}\nHrudhi: "
            
            # Generate response using the pipeline directly
            response = self.chat_pipeline(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.chat_tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract just the generated part (after the prompt)
            generated_part = response[len(prompt):].strip()
            
            # Clean up the response
            if generated_part:
                # Stop at natural break points
                stop_tokens = ['\nUser:', '\nHuman:', '\n\n', 'User:', 'Human:']
                for stop in stop_tokens:
                    if stop in generated_part:
                        generated_part = generated_part.split(stop)[0]
                
                # Add conversation to history
                self.conversation_history.append({
                    'user': user_message,
                    'assistant': generated_part,
                    'timestamp': datetime.now().isoformat()
                })
                
                return f"🤖 {generated_part}"
            else:
                return "🤖 I understand you're trying to chat with me. How can I help you today?"
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"🤖 I encountered an error while processing your request: {str(e)}\n\nLet me try to help you in a different way. What would you like to know?"
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Generate AI summary of text"""
        if not self.summarizer or not text.strip():
            return self._fallback_summary(text)
        
        try:
            # Clean text for summarization
            clean_text = re.sub(r'\n+', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 50:
                return clean_text  # Too short to summarize
            
            # Generate summary
            result = self.summarizer(
                clean_text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._fallback_summary(text)
    
    def _fallback_summary(self, text: str) -> str:
        """Simple fallback summary if AI fails"""
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text
        
        # Take first sentence and last if short enough
        summary = sentences[0] + '.'
        if len(summary) < 100 and len(sentences) > 1:
            summary += ' ' + sentences[1] + '.'
        
        return summary
    
    def search_notes(self, query: str, notes: List[Note], top_k: int = 5) -> List[tuple]:
        """Semantic search through notes"""
        if not self.embedder or not query.strip():
            return self._text_search(query, notes)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create search corpus
            corpus = []
            note_map = {}
            
            for note in notes:
                text = f"{note.title} {note.content} {note.summary} {' '.join(note.tags)}"
                corpus.append(text)
                note_map[len(corpus) - 1] = note
            
            if not corpus:
                return []
            
            # Calculate similarities
            query_embedding = self.embedder.encode([query])
            corpus_embeddings = self.embedder.encode(corpus)
            scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
            
            # Get top matches
            results = []
            for i, score in enumerate(scores):
                if score > 0.15:  # Similarity threshold
                    results.append((note_map[i], score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"AI search failed: {e}")
            return self._text_search(query, notes)
    
    def _text_search(self, query: str, notes: List[Note]) -> List[tuple]:
        """Fallback text search"""
        results = []
        query_lower = query.lower()
        
        for note in notes:
            score = 0
            if query_lower in note.title.lower():
                score += 0.8
            if query_lower in note.content.lower():
                score += 0.6
            if query_lower in note.summary.lower():
                score += 0.7
            if any(query_lower in tag.lower() for tag in note.tags):
                score += 0.9
            
            if score > 0:
                results.append((note, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]

class Robot3D:
    """3D Robotic avatar with animations"""
    
    def __init__(self, surface, x, y, size=80):
        self.surface = surface
        self.x = x
        self.y = y
        self.size = size
        self.animation_time = 0
        self.mood = "neutral"  # neutral, happy, thinking, working
        self.eye_blink = 0
        
    def update(self, dt):
        """Update robot animation"""
        self.animation_time += dt * 2  # Animation speed
        
        # Blinking animation
        if time.time() % 3 < 0.2:
            self.eye_blink = min(self.eye_blink + dt * 10, 1.0)
        else:
            self.eye_blink = max(self.eye_blink - dt * 10, 0.0)
    
    def set_mood(self, mood: str):
        """Set robot mood: neutral, happy, thinking, working"""
        self.mood = mood
    
    def draw(self):
        """Draw the 3D robot"""
        try:
            # Robot body (rounded rectangle with gradient)
            body_rect = pygame.Rect(self.x - self.size//2, self.y - self.size//2, self.size, self.size * 0.8)
            
            # Body gradient effect
            for i in range(20):
                alpha = 255 - i * 10
                color = (100 + i * 3, 150 + i * 2, 255 - i * 2, alpha)
                offset_rect = body_rect.inflate(-i, -i)
                if offset_rect.width > 0 and offset_rect.height > 0:
                    pygame.draw.ellipse(self.surface, color[:3], offset_rect)
            
            # Head
            head_y = self.y - self.size//2 - 20
            head_size = self.size * 0.6
            head_rect = pygame.Rect(self.x - head_size//2, head_y - head_size//2, head_size, head_size)
            
            # Head gradient
            for i in range(15):
                alpha = 255 - i * 12
                color = (120 + i * 4, 160 + i * 3, 255 - i * 3)
                offset_rect = head_rect.inflate(-i, -i)
                if offset_rect.width > 0 and offset_rect.height > 0:
                    pygame.draw.ellipse(self.surface, color, offset_rect)
            
            # Eyes
            eye_offset = math.sin(self.animation_time) * 2
            left_eye_x = self.x - 12
            right_eye_x = self.x + 12
            eye_y = head_y - 5 + eye_offset
            
            # Eye glow effect
            eye_size = 8 if self.eye_blink < 0.5 else 8 * (1 - self.eye_blink)
            
            if self.mood == "happy":
                # Happy eyes (curved)
                pygame.draw.arc(self.surface, (0, 255, 100), 
                               (left_eye_x - 6, eye_y - 3, 12, 8), 0, math.pi, 3)
                pygame.draw.arc(self.surface, (0, 255, 100), 
                               (right_eye_x - 6, eye_y - 3, 12, 8), 0, math.pi, 3)
            elif self.mood == "thinking":
                # Thinking eyes (looking up)
                pygame.draw.circle(self.surface, (100, 200, 255), (left_eye_x, eye_y - 2), int(eye_size))
                pygame.draw.circle(self.surface, (100, 200, 255), (right_eye_x, eye_y - 2), int(eye_size))
            else:
                # Normal eyes
                pygame.draw.circle(self.surface, (0, 200, 255), (left_eye_x, eye_y), int(eye_size))
                pygame.draw.circle(self.surface, (0, 200, 255), (right_eye_x, eye_y), int(eye_size))
            
            # Antenna with glowing tip
            antenna_x = self.x
            antenna_y = head_y - head_size//2 - 10
            pygame.draw.line(self.surface, (150, 150, 150), 
                            (antenna_x, head_y - head_size//2), (antenna_x, antenna_y), 2)
            
            # Glowing antenna tip
            glow_intensity = (math.sin(self.animation_time * 3) + 1) / 2
            tip_color = (int(255 * glow_intensity), int(200 * glow_intensity), 0)
            pygame.draw.circle(self.surface, tip_color, (antenna_x, antenna_y), 4)
            
            # Arms (animated)
            arm_swing = math.sin(self.animation_time * 1.5) * 10
            left_arm_end = (self.x - self.size//2 - 15, self.y + arm_swing)
            right_arm_end = (self.x + self.size//2 + 15, self.y - arm_swing)
            
            pygame.draw.line(self.surface, (100, 150, 200), 
                            (self.x - self.size//2, self.y), left_arm_end, 4)
            pygame.draw.line(self.surface, (100, 150, 200), 
                            (self.x + self.size//2, self.y), right_arm_end, 4)
            
            # Hand circles
            pygame.draw.circle(self.surface, (80, 130, 180), left_arm_end, 6)
            pygame.draw.circle(self.surface, (80, 130, 180), right_arm_end, 6)
            
        except Exception as e:
            # Fallback simple robot if pygame fails
            logger.warning(f"3D robot drawing failed: {e}")
            self._draw_simple_robot()
    
    def _draw_simple_robot(self):
        """Simple fallback robot drawing"""
        try:
            # Simple colored rectangles
            pygame.draw.rect(self.surface, (100, 150, 255), 
                           (self.x - 20, self.y - 30, 40, 50))
            pygame.draw.circle(self.surface, (120, 170, 255), 
                             (self.x, self.y - 40), 15)
            pygame.draw.circle(self.surface, (0, 200, 255), 
                             (self.x - 5, self.y - 42), 3)
            pygame.draw.circle(self.surface, (0, 200, 255), 
                             (self.x + 5, self.y - 42), 3)
        except:
            pass  # If all else fails, just skip drawing

class MarkdownEditor:
    """Markdown editor with live preview"""
    
    def __init__(self, parent):
        self.parent = parent
        self.preview_enabled = False
        
    def apply_markdown_formatting(self, text_widget):
        """Apply basic markdown highlighting to text widget"""
        try:
            content = text_widget.get(1.0, tk.END)
            
            # Configure text tags for markdown
            text_widget.tag_config("header", font=("Segoe UI", 14, "bold"), foreground="#4F46E5")
            text_widget.tag_config("bold", font=("Segoe UI", 11, "bold"))
            text_widget.tag_config("italic", font=("Segoe UI", 11, "italic"))
            text_widget.tag_config("code", font=("Consolas", 10), background="#F3F4F6")
            text_widget.tag_config("link", foreground="#2563EB", underline=True)
            
            # Remove existing tags
            for tag in ["header", "bold", "italic", "code", "link"]:
                text_widget.tag_remove(tag, 1.0, tk.END)
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                line_start = f"{i+1}.0"
                line_end = f"{i+1}.end"
                
                # Headers
                if line.startswith('#'):
                    text_widget.tag_add("header", line_start, line_end)
                
                # Bold text
                bold_matches = re.finditer(r'\*\*(.*?)\*\*', line)
                for match in bold_matches:
                    start_idx = f"{i+1}.{match.start()}"
                    end_idx = f"{i+1}.{match.end()}"
                    text_widget.tag_add("bold", start_idx, end_idx)
                
                # Italic text
                italic_matches = re.finditer(r'\*(.*?)\*', line)
                for match in italic_matches:
                    if not re.match(r'\*\*(.*?)\*\*', match.group(0)):  # Not bold
                        start_idx = f"{i+1}.{match.start()}"
                        end_idx = f"{i+1}.{match.end()}"
                        text_widget.tag_add("italic", start_idx, end_idx)
                
                # Code blocks
                code_matches = re.finditer(r'`(.*?)`', line)
                for match in code_matches:
                    start_idx = f"{i+1}.{match.start()}"
                    end_idx = f"{i+1}.{match.end()}"
                    text_widget.tag_add("code", start_idx, end_idx)
                
                # Links
                link_matches = re.finditer(r'\[([^\]]+)\]\([^\)]+\)', line)
                for match in link_matches:
                    start_idx = f"{i+1}.{match.start()}"
                    end_idx = f"{i+1}.{match.end()}"
                    text_widget.tag_add("link", start_idx, end_idx)
                    
        except Exception as e:
            logger.error(f"Markdown formatting error: {e}")

class HrudhiPersonalAssistant:
    """Main Hrudhi Personal Assistant application"""
    
    def __init__(self):
        # Initialize pygame for 3D robot
        pygame.init()
        
        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        # Main window with glassmorphism
        self.root = ctk.CTk()
        self.root.title("🤖 Hrudhi Personal Assistant")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Data
        self.notes: List[Note] = []
        self.filtered_notes: List[Note] = []
        self.selected_note: Optional[Note] = None
        self.data_file = Path("hrudhi_assistant_data.json")
        self.current_theme = "system"
        
        # Initialize chat state
        self.chat_visible = False
        
        # AI and components
        self.ai_engine = AIEngine()
        self.markdown_editor = MarkdownEditor(self)
        
        # Check actual AI availability after engine initialization
        self.ai_available = self.ai_engine.chat_pipeline is not None
        
        # Robot avatar setup
        self.setup_robot_canvas()
        
        # UI variables
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search_changed)
        
        # Theme variables
        self.theme_var = tk.StringVar(value="system")
        
        # Initialize UI
        self.setup_modern_ui()
        self.load_notes()
        self.refresh_notes_list()
        self.setup_autosave()
        
        # Start robot animation
        self.start_robot_animation()
        
    def setup_robot_canvas(self):
        """Setup pygame canvas for 3D robot"""
        try:
            # Create pygame surface
            self.robot_surface = pygame.Surface((120, 120), pygame.SRCALPHA)
            self.robot = Robot3D(self.robot_surface, 60, 70, 60)
        except Exception as e:
            logger.warning(f"Robot canvas setup failed: {e}")
            self.robot_surface = None
            self.robot = None
    
    def start_robot_animation(self):
        """Start robot animation thread"""
        if self.robot:
            def animate_robot():
                clock = pygame.time.Clock()
                while True:
                    try:
                        dt = clock.tick(30) / 1000.0  # 30 FPS
                        self.robot.update(dt)
                        
                        # Clear surface
                        self.robot_surface.fill((0, 0, 0, 0))
                        self.robot.draw()
                        
                        # Convert pygame surface to PIL Image and then to PhotoImage
                        # This would need additional conversion code for tkinter integration
                        
                        time.sleep(0.033)  # ~30 FPS
                    except Exception as e:
                        logger.error(f"Robot animation error: {e}")
                        time.sleep(1)
            
            robot_thread = threading.Thread(target=animate_robot, daemon=True)
            robot_thread.start()
    
    def setup_modern_ui(self):
        """Setup modern UI with glassmorphism effects"""
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main layout
        self.create_glassmorphism_sidebar()
        self.create_main_content_area()
        self.create_robot_panel()
        self.create_modern_status_bar()
    
    def create_glassmorphism_sidebar(self):
        """Create left sidebar with glassmorphism effects"""
        # Sidebar with rounded corners and subtle shadow
        self.sidebar = ctk.CTkFrame(
            self.root, 
            width=320,
            corner_radius=20,
            fg_color=("#F8FAFC", "#1E293B"),
            border_width=1,
            border_color=("#E2E8F0", "#334155")
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(15, 8), pady=15)
        self.sidebar.grid_propagate(False)
        
        # Title with gradient effect
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 15))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🤖 Hrudhi Assistant",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=("#4F46E5", "#818CF8")
        )
        title_label.grid(row=0, column=0, sticky="w")
        
        # Theme selector
        theme_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        theme_frame.grid(row=0, column=1, sticky="e")
        
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["🌙 Dark", "☀️ Light", "🔄 Auto"],
            command=self.change_theme,
            width=100,
            height=28,
            corner_radius=15
        )
        theme_menu.grid(row=0, column=0)
        theme_menu.set("🔄 Auto")
        
        # Enhanced search bar with AI indicator
        search_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        search_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
        search_frame.grid_columnconfigure(0, weight=1)
        
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="🔍 AI-powered search...",
            textvariable=self.search_var,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=12)
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        
        clear_btn = ctk.CTkButton(
            search_frame,
            text="✕",
            width=40,
            height=40,
            corner_radius=20,
            command=self.clear_search,
            font=ctk.CTkFont(size=14)
        )
        clear_btn.grid(row=0, column=1)
        
        # Control buttons with modern styling
        controls_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        controls_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
        controls_frame.grid_columnconfigure((0, 1), weight=1)
        
        new_btn = ctk.CTkButton(
            controls_frame,
            text="✨ New Note",
            command=self.create_new_note,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#6366F1", "#4F46E5"),
            hover_color=("#4F46E5", "#4338CA")
        )
        new_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        
        import_btn = ctk.CTkButton(
            controls_frame,
            text="📁 Import",
            command=self.import_notes,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=12)
        )
        import_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        
        # Chat assistant button
        # AI Chat toggle with modern styling
        chat_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        chat_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 15))
        
        self.chat_open = False
        self.chat_btn = ctk.CTkButton(
            chat_frame,
            text="🤖 AI Chat" if self.ai_available else "🤖 AI Chat (Unavailable)",
            command=self.toggle_chat_panel if self.ai_available else self.show_ai_unavailable_message,
            height=45,
            corner_radius=20,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#10B981", "#059669") if self.ai_available else ("#6B7280", "#4B5563"),
            hover_color=("#059669", "#047857") if self.ai_available else ("#4B5563", "#374151"),
            state="normal" if self.ai_available else "disabled"
        )
        self.chat_btn.pack(fill=tk.X)
        
        # Learning and Link buttons
        learning_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        learning_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 15))
        learning_frame.grid_columnconfigure((0, 1), weight=1)
        
        learn_btn = ctk.CTkButton(
            learning_frame,
            text="🧠 Teach AI" if self.ai_available else "🧠 AI Learning (Unavailable)",
            command=self.open_learning_dialog if self.ai_available else self.show_ai_unavailable_message,
            height=35,
            corner_radius=15,
            font=ctk.CTkFont(size=11),
            fg_color=("#8B5CF6", "#7C3AED") if self.ai_available else ("#6B7280", "#4B5563"),
            hover_color=("#7C3AED", "#6D28D9") if self.ai_available else ("#4B5563", "#374151"),
            state="normal" if self.ai_available else "disabled"
        )
        learn_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        
        link_btn = ctk.CTkButton(
            learning_frame,
            text="🔗 Add Link" if self.ai_available else "🔗 Links (Unavailable)",
            command=self.open_link_dialog if self.ai_available else self.show_ai_unavailable_message,
            height=35,
            corner_radius=15,
            font=ctk.CTkFont(size=11),
            fg_color=("#F59E0B", "#D97706") if self.ai_available else ("#6B7280", "#4B5563"),
            hover_color=("#D97706", "#B45309") if self.ai_available else ("#4B5563", "#374151"),
            state="normal" if self.ai_available else "disabled"
        )
        link_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        
        # Notes list with enhanced styling
        self.create_enhanced_notes_list()
        
        # Configure sidebar grid
        self.sidebar.grid_rowconfigure(5, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
    
    def create_enhanced_notes_list(self):
        """Create enhanced notes list with modern styling"""
        list_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        list_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=(0, 20))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Custom listbox with modern styling
        self.notes_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            selectmode=tk.SINGLE,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            bg=("#FFFFFF", "#2D2D2D")[ctk.get_appearance_mode() == "Dark"],
            fg=("#1F2937", "#E5E7EB")[ctk.get_appearance_mode() == "Dark"],
            selectbackground="#6366F1",
            selectforeground="#FFFFFF",
            relief="flat"
        )
        self.notes_listbox.grid(row=0, column=0, sticky="nsew")
        self.notes_listbox.bind('<<ListboxSelect>>', self.on_note_selected)
        self.notes_listbox.bind('<Double-Button-1>', self.on_note_double_click)
        
        # Custom scrollbar
        scrollbar = ctk.CTkScrollbar(
            list_frame,
            orientation="vertical",
            command=self.notes_listbox.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(5, 0))
        self.notes_listbox.configure(yscrollcommand=scrollbar.set)
    
    def create_main_content_area(self):
        """Create main content area with glassmorphism and chat integration"""
        # Main container that can switch between notes and chat
        self.main_container = ctk.CTkFrame(
            self.root,
            corner_radius=20,
            fg_color=("#FFFFFF", "#1E293B"),
            border_width=1,
            border_color=("#E2E8F0", "#334155")
        )
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=(8, 15), pady=15)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Notes editor frame (default view)
        self.notes_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.notes_frame.grid(row=0, column=0, sticky="nsew")
        self.notes_frame.grid_rowconfigure(3, weight=1)
        self.notes_frame.grid_columnconfigure(0, weight=1)
        
        # Create notes interface
        self.create_notes_interface()
        
        # Chat frame (hidden by default)
        self.chat_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        # Don't grid it yet - will be shown when chat is toggled
        
        # Create chat interface
        self.create_chat_interface()
    
    def create_notes_interface(self):
        """Create the notes editing interface"""
        # Note title with enhanced styling
        self.title_entry = ctk.CTkEntry(
            self.notes_frame,
            placeholder_text="✏️ Note title...",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=45,
            corner_radius=15
        )
        self.title_entry.grid(row=0, column=0, sticky="ew", padx=25, pady=(25, 15))
        self.title_entry.bind('<KeyRelease>', self.on_content_changed)
        
        # Enhanced toolbar
        self.create_enhanced_toolbar(self.notes_frame)
        
        # AI Summary section
        self.create_ai_summary_section(self.notes_frame)
        
        # Content editor with markdown support
        self.create_content_editor(self.notes_frame)
    
    def create_chat_interface(self):
        """Create integrated AI chat interface"""
        self.chat_frame.grid_rowconfigure(1, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Chat header
        header_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent", height=60)
        header_frame.grid(row=0, column=0, sticky="ew", padx=25, pady=(25, 15))
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        chat_title = ctk.CTkLabel(
            header_frame,
            text="🤖 Hrudhi AI Chat Assistant",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#4F46E5", "#818CF8")
        )
        chat_title.grid(row=0, column=0, sticky="w", pady=10)
        
        # Chat controls
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.grid(row=0, column=2, sticky="e", pady=10)
        
        clear_chat_btn = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear",
            command=self.clear_chat,
            width=80,
            height=30,
            corner_radius=15,
            font=ctk.CTkFont(size=10),
            fg_color=("#EF4444", "#DC2626"),
            hover_color=("#DC2626", "#B91C1C")
        )
        clear_chat_btn.pack(side="right", padx=(10, 0))
        
        # Chat display area
        chat_display_frame = ctk.CTkFrame(
            self.chat_frame,
            corner_radius=15,
            fg_color=("#F8FAFC", "#0F172A")
        )
        chat_display_frame.grid(row=1, column=0, sticky="nsew", padx=25, pady=(0, 15))
        chat_display_frame.grid_rowconfigure(0, weight=1)
        chat_display_frame.grid_columnconfigure(0, weight=1)
        
        # Chat messages scrollable text
        self.chat_display = scrolledtext.ScrolledText(
            chat_display_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg=("#FFFFFF", "#1E293B")[ctk.get_appearance_mode() == "Dark"],
            fg=("#1F2937", "#E5E7EB")[ctk.get_appearance_mode() == "Dark"],
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=15,
            state="disabled"
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        # Input area
        input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent", height=80)
        input_frame.grid(row=2, column=0, sticky="ew", padx=25, pady=(0, 25))
        input_frame.grid_propagate(False)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Chat input
        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="💬 Ask me anything about your notes or share links to learn from...",
            font=ctk.CTkFont(size=12),
            height=40,
            corner_radius=20
        )
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_input.bind('<Return>', self.send_chat_message)
        self.chat_input.bind('<Control-Return>', lambda e: self.chat_input.insert(tk.END, '\n'))
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="🚀 Send",
            command=self.send_chat_message,
            width=80,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#10B981", "#059669"),
            hover_color=("#059669", "#047857")
        )
        self.send_btn.grid(row=0, column=1)
        
        # Initialize chat
        self.add_chat_message("🤖 Hi! I'm Hrudhi, your AI assistant. I know about all your notes and can learn from new information you share. How can I help you today?", "assistant")
    
    def toggle_chat_panel(self):
        """Toggle between notes view and chat view"""
        if hasattr(self, 'chat_visible') and self.chat_visible:
            # Switch to notes view
            self.chat_frame.grid_remove()
            self.notes_frame.grid(row=0, column=0, sticky="nsew")
            self.chat_btn.configure(text="💬 Chat")
            self.chat_visible = False
        else:
            # Switch to chat view
            self.notes_frame.grid_remove()
            self.chat_frame.grid(row=0, column=0, sticky="nsew")
            self.chat_frame.grid_rowconfigure(1, weight=1)
            self.chat_frame.grid_columnconfigure(0, weight=1)
            self.chat_btn.configure(text="📝 Notes")
            self.chat_visible = True
    
    def send_chat_message(self, event=None):
        """Send message to AI chat"""
        message = self.chat_input.get().strip()
        if not message:
            return
            
        # Add user message
        self.add_chat_message(message, "user")
        self.chat_input.delete(0, tk.END)
        
        # Disable send button and show thinking
        self.send_btn.configure(state="disabled", text="🤔 Thinking...")
        self.root.update()
        
        try:
            # Get AI response
            response = self.ai_engine.generate_chat_response(message)
            self.add_chat_message(response, "assistant")
        except Exception as e:
            self.add_chat_message(f"🚫 Sorry, I encountered an error: {str(e)}", "assistant")
        finally:
            # Re-enable send button
            self.send_btn.configure(state="normal", text="🚀 Send")
    
    def add_chat_message(self, message, sender):
        """Add message to chat display"""
        self.chat_display.configure(state="normal")
        
        # Add timestamp and sender
        timestamp = datetime.now().strftime("%H:%M")
        if sender == "user":
            prefix = f"[{timestamp}] 👤 You: "
            self.chat_display.insert(tk.END, prefix, "user_label")
        else:
            prefix = f"[{timestamp}] 🤖 Hrudhi: "
            self.chat_display.insert(tk.END, prefix, "bot_label")
        
        # Add message content
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Configure text tags for styling
        self.chat_display.tag_config("user_label", foreground="#10B981", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("bot_label", foreground="#6366F1", font=("Segoe UI", 10, "bold"))
        
        # Auto-scroll to bottom
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear chat history"""
        result = messagebox.askyesno("Clear Chat", "🗑️ Clear all chat messages?")
        if result:
            self.chat_display.configure(state="normal")
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.configure(state="disabled")
            # Reset AI conversation history
            self.ai_engine.conversation_history = []
            # Add welcome message back
            self.add_chat_message("🤖 Chat cleared! How can I help you today?", "assistant")
    
    def open_learning_dialog(self):
        """Open dialog to teach AI new information"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("🧠 Teach Hrudhi AI")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Dialog content
        dialog.grid_rowconfigure(1, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            dialog,
            text="🧠 Teach Hrudhi New Information",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#4F46E5", "#818CF8")
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # Instructions
        instructions = ctk.CTkLabel(
            dialog,
            text="Share any information you'd like Hrudhi to remember and use in future conversations:",
            font=ctk.CTkFont(size=12),
            wraplength=550
        )
        instructions.grid(row=1, column=0, pady=(0, 15), padx=20)
        
        # Text input
        text_frame = ctk.CTkFrame(dialog)
        text_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        learning_text = ctk.CTkTextbox(
            text_frame,
            height=200,
            font=ctk.CTkFont(size=11),
            wrap="word"
        )
        learning_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(row=3, column=0, pady=20)
        
        def save_learning():
            content = learning_text.get("1.0", tk.END).strip()
            if content:
                self.ai_engine.add_learning_data(content)
                messagebox.showinfo("Success", "🧠 Information added to Hrudhi's knowledge base!")
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please enter some information to teach.")
        
        save_btn = ctk.CTkButton(
            button_frame,
            text="🧠 Teach Hrudhi",
            command=save_learning,
            width=120,
            fg_color=("#10B981", "#059669"),
            hover_color=("#059669", "#047857")
        )
        save_btn.pack(side="left", padx=(0, 10))
        
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ Cancel",
            command=dialog.destroy,
            width=100,
            fg_color=("#6B7280", "#4B5563"),
            hover_color=("#4B5563", "#374151")
        )
        cancel_btn.pack(side="left")
        
        learning_text.focus()
    
    def open_link_dialog(self):
        """Open dialog to add learning links"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("🔗 Add Learning Link")
        dialog.geometry("600x450")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Dialog content
        dialog.grid_rowconfigure(2, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            dialog,
            text="🔗 Share Learning Links with Hrudhi",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#4F46E5", "#818CF8")
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # URL input frame
        url_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        url_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
        url_frame.grid_columnconfigure(1, weight=1)
        
        url_label = ctk.CTkLabel(url_frame, text="URL:", font=ctk.CTkFont(size=12, weight="bold"))
        url_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        url_entry = ctk.CTkEntry(
            url_frame,
            placeholder_text="https://example.com/article",
            font=ctk.CTkFont(size=11),
            height=35
        )
        url_entry.grid(row=0, column=1, sticky="ew")
        
        # Description input frame
        desc_frame = ctk.CTkFrame(dialog)
        desc_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        desc_frame.grid_rowconfigure(1, weight=1)
        desc_frame.grid_columnconfigure(0, weight=1)
        
        desc_label = ctk.CTkLabel(desc_frame, text="📝 Optional Description:", font=ctk.CTkFont(size=12, weight="bold"))
        desc_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        desc_text = ctk.CTkTextbox(
            desc_frame,
            height=120,
            font=ctk.CTkFont(size=11),
            wrap="word"
        )
        desc_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Add placeholder-like text
        placeholder = "Briefly describe what this link contains or why it's useful..."
        desc_text.insert("1.0", placeholder)
        desc_text.configure(text_color=("#9CA3AF", "#6B7280"))  # Gray color for placeholder
        
        def on_desc_focus_in(event):
            if desc_text.get("1.0", "end-1c") == placeholder:
                desc_text.delete("1.0", "end")
                desc_text.configure(text_color=("black", "white"))  # Normal text color
        
        def on_desc_focus_out(event):
            if not desc_text.get("1.0", "end-1c").strip():
                desc_text.insert("1.0", placeholder)
                desc_text.configure(text_color=("#9CA3AF", "#6B7280"))  # Gray color for placeholder
        
        desc_text.bind("<FocusIn>", on_desc_focus_in)
        desc_text.bind("<FocusOut>", on_desc_focus_out)
        
        # Progress label (hidden initially)
        progress_label = ctk.CTkLabel(
            dialog,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("#10B981", "#059669")
        )
        progress_label.grid(row=3, column=0, pady=(0, 10))
        
        # Status label for feedback
        status_label = ctk.CTkLabel(
            dialog,
            text="💡 Tip: Paste a Dynatrace documentation URL and add a description",
            font=ctk.CTkFont(size=10),
            text_color=("#6B7280", "#9CA3AF")
        )
        status_label.grid(row=4, column=0, pady=(0, 10))
        
        # Button frame
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(row=5, column=0, pady=20)
        
        def add_link():
            """Process and add the link to AI knowledge base"""
            url = url_entry.get().strip()
            description = desc_text.get("1.0", "end-1c").strip()
            
            # Remove placeholder text if it's still there
            if description == placeholder:
                description = ""
            
            if not url:
                messagebox.showwarning("Missing URL", "Please enter a URL to add.")
                url_entry.focus()
                return
            
            # Basic URL validation
            if not (url.startswith('http://') or url.startswith('https://')):
                if messagebox.askyesno("Fix URL?", f"URL should start with http:// or https://\n\nAdd https:// to: {url}?"):
                    url = f"https://{url}"
                    url_entry.delete(0, "end")
                    url_entry.insert(0, url)
                else:
                    return
            
            try:
                # Show progress
                progress_label.configure(text="🔄 Fetching content from URL...")
                status_label.configure(text="")
                dialog.update()
                
                # Add to AI knowledge base
                success = self.ai_engine.add_link_data(url, title=description)
                
                if success:
                    progress_label.configure(text="✅ Successfully added to knowledge base!")
                    messagebox.showinfo(
                        "Link Added Successfully", 
                        f"🔗 Link added to Hrudhi's knowledge base!\n\n"
                        f"URL: {url}\n"
                        f"{'Description: ' + description if description else ''}\n\n"
                        f"Hrudhi can now answer questions about this content in chat!"
                    )
                    dialog.destroy()
                else:
                    progress_label.configure(text="❌ Failed to process link")
                    messagebox.showerror("Error", "Failed to process the link. Please check the URL and try again.")
            
            except Exception as e:
                progress_label.configure(text="❌ Error occurred")
                messagebox.showerror("Error", f"Failed to process link: {str(e)}")
        
        def paste_url():
            """Paste URL from clipboard"""
            try:
                clipboard_content = dialog.clipboard_get()
                if clipboard_content and (clipboard_content.startswith('http://') or clipboard_content.startswith('https://')):
                    url_entry.delete(0, "end")
                    url_entry.insert(0, clipboard_content)
                    desc_text.focus()
                else:
                    messagebox.showinfo("Clipboard", "No valid URL found in clipboard.")
            except:
                messagebox.showwarning("Clipboard", "Could not access clipboard.")
        
        # Buttons
        paste_btn = ctk.CTkButton(
            button_frame,
            text="📋 Paste URL",
            command=paste_url,
            width=100,
            fg_color=("#8B5CF6", "#7C3AED"),
            hover_color=("#7C3AED", "#6D28D9")
        )
        paste_btn.pack(side="left", padx=(0, 10))
        
        add_btn = ctk.CTkButton(
            button_frame,
            text="🔗 Add Link",
            command=add_link,
            width=120,
            fg_color=("#3B82F6", "#2563EB"),
            hover_color=("#2563EB", "#1D4ED8")
        )
        add_btn.pack(side="left", padx=(0, 10))
        
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ Cancel",
            command=dialog.destroy,
            width=100,
            fg_color=("#6B7280", "#4B5563"),
            hover_color=("#4B5563", "#374151")
        )
        cancel_btn.pack(side="left")
        
        # Key bindings for better UX
        def on_enter(event):
            if event.widget == url_entry:
                desc_text.focus()
            elif event.widget == desc_text:
                add_link()
        
        url_entry.bind('<Return>', on_enter)
        dialog.bind('<Control-Return>', lambda e: add_link())
        
        # Focus on URL entry
        url_entry.focus()
    
    def show_ai_unavailable_message(self):
        """Show message when AI features are not available"""
        messagebox.showinfo(
            "AI Unavailable",
            "🤖 AI chat features are not available.\n\n"
            "This could be due to:\n"
            "• Missing AI dependencies\n"
            "• Model loading failed\n"
            "• System compatibility issues\n\n"
            "You can still use all the note-taking features!"
        )
    
    def create_enhanced_toolbar(self, parent):
        """Create enhanced toolbar with AI features"""
        toolbar = ctk.CTkFrame(parent, height=60, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=25, pady=(0, 15))
        toolbar.grid_propagate(False)
        
        # Left side buttons
        left_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        left_frame.pack(side="left", fill="y")
        
        save_btn = ctk.CTkButton(
            left_frame,
            text="💾 Save",
            command=self.save_current_note,
            width=90,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=11)
        )
        save_btn.pack(side="left", padx=(0, 10))
        
        # AI Summarize button (highlighted)
        self.summarize_btn = ctk.CTkButton(
            left_frame,
            text="🧠 AI Summary",
            command=self.generate_ai_summary,
            width=110,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#10B981", "#059669"),
            hover_color=("#059669", "#047857")
        )
        self.summarize_btn.pack(side="left", padx=(0, 10))
        
        pin_btn = ctk.CTkButton(
            left_frame,
            text="📌 Pin",
            command=self.toggle_pin_note,
            width=80,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=11)
        )
        pin_btn.pack(side="left", padx=(0, 10))
        
        # Markdown toggle
        self.markdown_var = tk.BooleanVar()
        markdown_switch = ctk.CTkSwitch(
            left_frame,
            text="📝 Markdown",
            variable=self.markdown_var,
            command=self.toggle_markdown_mode,
            font=ctk.CTkFont(size=11)
        )
        markdown_switch.pack(side="left", padx=(0, 10))
        
        # Right side buttons
        right_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        right_frame.pack(side="right", fill="y")
        
        delete_btn = ctk.CTkButton(
            right_frame,
            text="🗑️ Delete",
            command=self.delete_current_note,
            width=90,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=11),
            fg_color=("#DC2626", "#B91C1C"),
            hover_color=("#B91C1C", "#991B1B")
        )
        delete_btn.pack(side="right", padx=(10, 0))
        
        export_btn = ctk.CTkButton(
            right_frame,
            text="📤 Export",
            command=self.export_notes,
            width=90,
            height=35,
            corner_radius=18,
            font=ctk.CTkFont(size=11)
        )
        export_btn.pack(side="right", padx=(10, 0))
    
    def create_ai_summary_section(self, parent):
        """Create AI summary display section"""
        summary_frame = ctk.CTkFrame(
            parent,
            height=80,
            corner_radius=15,
            fg_color=("#F0F4FF", "#1E1B4B"),
            border_width=1,
            border_color=("#6366F1", "#818CF8")
        )
        summary_frame.grid(row=2, column=0, sticky="ew", padx=25, pady=(0, 15))
        summary_frame.grid_propagate(False)
        summary_frame.grid_columnconfigure(0, weight=1)
        
        # Summary label
        summary_label = ctk.CTkLabel(
            summary_frame,
            text="🧠 AI Summary",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#6366F1", "#818CF8")
        )
        summary_label.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))
        
        # Summary text
        self.summary_text = ctk.CTkTextbox(
            summary_frame,
            height=40,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            border_width=0
        )
        self.summary_text.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        self.summary_text.configure(state="disabled")
    
    def create_content_editor(self, parent):
        """Create enhanced content editor"""
        content_frame = ctk.CTkFrame(
            parent,
            corner_radius=15,
            fg_color="transparent"
        )
        content_frame.grid(row=3, column=0, sticky="nsew", padx=25, pady=(0, 25))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Enhanced text editor
        self.content_text = tk.Text(
            content_frame,
            font=("Consolas", 11),
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=1,
            highlightcolor="#6366F1",
            bg=("#FFFFFF", "#2D2D2D")[ctk.get_appearance_mode() == "Dark"],
            fg=("#1F2937", "#E5E7EB")[ctk.get_appearance_mode() == "Dark"],
            insertbackground=("#1F2937", "#E5E7EB")[ctk.get_appearance_mode() == "Dark"],
            padx=20,
            pady=20,
            relief="flat"
        )
        self.content_text.grid(row=0, column=0, sticky="nsew")
        self.content_text.bind('<KeyRelease>', self.on_content_changed)
        self.content_text.bind('<Key>', self.on_markdown_key)
        
        # Enhanced scrollbar
        content_scrollbar = ctk.CTkScrollbar(
            content_frame,
            orientation="vertical",
            command=self.content_text.yview
        )
        content_scrollbar.grid(row=0, column=1, sticky="ns", padx=(5, 0))
        self.content_text.configure(yscrollcommand=content_scrollbar.set)
    
    def create_robot_panel(self):
        """Create robot avatar panel"""
        robot_frame = ctk.CTkFrame(
            self.root,
            width=140,
            height=200,
            corner_radius=20,
            fg_color=("#F8FAFC", "#1E293B"),
            border_width=1,
            border_color=("#E2E8F0", "#334155")
        )
        robot_frame.grid(row=0, column=2, sticky="ne", padx=(8, 15), pady=15)
        robot_frame.grid_propagate(False)
        
        # Robot avatar placeholder (would integrate pygame surface here)
        robot_label = ctk.CTkLabel(
            robot_frame,
            text="🤖\nHrudhi\nAssistant",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#6366F1", "#818CF8"),
            justify="center"
        )
        robot_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Status indicator
        self.robot_status = ctk.CTkLabel(
            robot_frame,
            text="💭 Ready",
            font=ctk.CTkFont(size=10),
            text_color=("#10B981", "#34D399")
        )
        self.robot_status.pack(pady=(0, 10))
    
    def create_modern_status_bar(self):
        """Create modern status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to assist you! ✨")
        
        status_frame = ctk.CTkFrame(
            self.root,
            height=35,
            corner_radius=15,
            fg_color=("#F1F5F9", "#1E293B")
        )
        status_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=15, pady=(0, 15))
        status_frame.grid_propagate(False)
        
        status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11),
            text_color=("#6B7280", "#9CA3AF")
        )
        status_label.pack(side="left", padx=20, pady=8)
        
        # Add note count
        self.note_count_var = tk.StringVar()
        note_count_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.note_count_var,
            font=ctk.CTkFont(size=10),
            text_color=("#6B7280", "#9CA3AF")
        )
        note_count_label.pack(side="right", padx=20, pady=8)
        
    # AI and functionality methods
    def generate_ai_summary(self):
        """Generate AI summary of current note"""
        if not self.selected_note:
            messagebox.showwarning("No Note", "Please select a note to summarize.")
            return
        
        content = self.content_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Empty Note", "Note content is empty.")
            return
        
        # Update robot status
        if self.robot:
            self.robot.set_mood("thinking")
        self.robot_status.configure(text="🧠 Thinking...")
        self.summarize_btn.configure(text="⏳ Processing...", state="disabled")
        
        def summarize_thread():
            try:
                summary = self.ai_engine.summarize_text(content, max_length=100)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.update_summary_ui(summary))
                
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                self.root.after(0, lambda: self.update_summary_ui(f"Summary generation failed: {str(e)}"))
        
        # Run summarization in background
        threading.Thread(target=summarize_thread, daemon=True).start()
    
    def update_summary_ui(self, summary):
        """Update summary UI with generated summary"""
        # Update summary display
        self.summary_text.configure(state="normal")
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        self.summary_text.configure(state="disabled")
        
        # Update note data
        if self.selected_note:
            self.selected_note.summary = summary
            self.save_current_note(show_status=False)
        
        # Reset UI
        if self.robot:
            self.robot.set_mood("happy")
        self.robot_status.configure(text="✨ Summary ready!")
        self.summarize_btn.configure(text="🧠 AI Summary", state="normal")
        self.status_var.set("AI summary generated successfully! 🧠✨")
    
    def toggle_markdown_mode(self):
        """Toggle markdown editing mode"""
        is_markdown = self.markdown_var.get()
        
        if is_markdown:
            self.markdown_editor.apply_markdown_formatting(self.content_text)
            self.status_var.set("Markdown mode enabled 📝")
        else:
            # Remove markdown formatting
            for tag in ["header", "bold", "italic", "code", "link"]:
                self.content_text.tag_remove(tag, 1.0, tk.END)
            self.status_var.set("Plain text mode enabled")
        
        # Update selected note
        if self.selected_note:
            self.selected_note.is_markdown = is_markdown
    
    def on_markdown_key(self, event):
        """Handle markdown key events"""
        if self.markdown_var.get():
            # Apply formatting after a short delay
            self.root.after(100, lambda: self.markdown_editor.apply_markdown_formatting(self.content_text))
    
    def change_theme(self, theme_choice):
        """Change application theme"""
        theme_map = {
            "🌙 Dark": "dark",
            "☀️ Light": "light", 
            "🔄 Auto": "system"
        }
        
        new_theme = theme_map.get(theme_choice, "system")
        ctk.set_appearance_mode(new_theme)
        self.current_theme = new_theme
        self.status_var.set(f"Theme changed to {theme_choice}")
        
        # Update robot mood
        if self.robot:
            self.robot.set_mood("happy")
    
    # Core note management methods (similar to before but enhanced)
    def create_new_note(self):
        """Create a new note"""
        note = Note(
            id=str(uuid.uuid4()),
            title="New Note",
            content="",
            summary="",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=[],
            color="#6366F1"
        )
        
        self.notes.append(note)
        self.refresh_notes_list()
        self.select_note(note)
        self.title_entry.focus()
        self.status_var.set("New note created ✨")
        
        # Update robot mood
        if self.robot:
            self.robot.set_mood("happy")
    
    def select_note(self, note: Note):
        """Select and display a note"""
        if self.selected_note and self.has_unsaved_changes():
            self.save_current_note()
        
        self.selected_note = note
        
        # Update UI
        self.title_entry.delete(0, tk.END)
        self.title_entry.insert(0, note.title)
        
        self.content_text.delete(1.0, tk.END)
        self.content_text.insert(1.0, note.content)
        
        # Update summary
        self.summary_text.configure(state="normal")
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, note.summary)
        self.summary_text.configure(state="disabled")
        
        # Update markdown mode
        self.markdown_var.set(note.is_markdown)
        if note.is_markdown:
            self.markdown_editor.apply_markdown_formatting(self.content_text)
        
        # Highlight in list
        try:
            index = self.filtered_notes.index(note)
            self.notes_listbox.selection_clear(0, tk.END)
            self.notes_listbox.selection_set(index)
            self.notes_listbox.see(index)
        except ValueError:
            pass
        
        # Update robot mood
        if self.robot:
            self.robot.set_mood("neutral")
    
    def save_current_note(self, show_status=True):
        """Save the currently selected note"""
        if not self.selected_note:
            return
        
        # Update note data
        self.selected_note.title = self.title_entry.get() or "Untitled"
        self.selected_note.content = self.content_text.get(1.0, tk.END).rstrip()
        self.selected_note.updated_at = datetime.now().isoformat()
        self.selected_note.is_markdown = self.markdown_var.get()
        
        # Generate tags from content
        self.selected_note.tags = self.extract_tags(self.selected_note.content)
        
        # Save to file
        self.save_notes()
        
        # Refresh display
        self.refresh_notes_list()
        
        if show_status:
            self.status_var.set("Note saved successfully! 💾")
            
        # Update robot mood
        if self.robot:
            self.robot.set_mood("happy")
    
    def extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content"""
        tags = re.findall(r'#(\w+)', content)
        return list(set(tags))  # Remove duplicates
    
    def on_note_selected(self, event):
        """Handle note selection from list"""
        selection = self.notes_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.filtered_notes):
                note = self.filtered_notes[index]
                self.select_note(note)
    
    def on_note_double_click(self, event):
        """Handle double-click on note"""
        self.title_entry.focus()
    
    def on_content_changed(self, event=None):
        """Handle content changes"""
        if self.selected_note:
            self.status_var.set("Modified ✏️")
            
            # Apply markdown formatting if enabled
            if self.markdown_var.get():
                self.root.after(500, lambda: self.markdown_editor.apply_markdown_formatting(self.content_text))
    
    def has_unsaved_changes(self) -> bool:
        """Check if current note has unsaved changes"""
        if not self.selected_note:
            return False
        
        current_title = self.title_entry.get()
        current_content = self.content_text.get(1.0, tk.END).rstrip()
        
        return (current_title != self.selected_note.title or 
                current_content != self.selected_note.content)
    
    def delete_current_note(self):
        """Delete the currently selected note"""
        if not self.selected_note:
            return
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Delete Note",
            f"Are you sure you want to delete '{self.selected_note.title}'?"
        )
        
        if result:
            self.notes.remove(self.selected_note)
            self.save_notes()
            self.refresh_notes_list()
            
            # Clear editor
            self.selected_note = None
            self.title_entry.delete(0, tk.END)
            self.content_text.delete(1.0, tk.END)
            self.summary_text.configure(state="normal")
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.configure(state="disabled")
            
            self.status_var.set("Note deleted 🗑️")
            
            # Update robot mood
            if self.robot:
                self.robot.set_mood("neutral")
    
    def toggle_pin_note(self):
        """Toggle pin status of current note"""
        if not self.selected_note:
            return
        
        self.selected_note.pinned = not self.selected_note.pinned
        self.save_current_note()
        self.refresh_notes_list()
        
        status = "pinned 📌" if self.selected_note.pinned else "unpinned"
        self.status_var.set(f"Note {status}")
    
    def on_search_changed(self, *args):
        """Handle search input changes"""
        query = self.search_var.get().strip()
        
        if not query:
            self.filtered_notes = self.notes.copy()
            self.status_var.set("Ready to assist you! ✨")
        else:
            # Update robot mood
            if self.robot:
                self.robot.set_mood("thinking")
            self.robot_status.configure(text="🔍 Searching...")
            
            # AI search
            ai_results = self.ai_engine.search_notes(query, self.notes)
            
            if ai_results:
                self.filtered_notes = [note for note, score in ai_results]
                self.status_var.set(f"🧠 AI found {len(ai_results)} relevant notes")
            else:
                self.filtered_notes = []
                self.status_var.set("No matching notes found")
            
            # Update robot status
            self.robot_status.configure(text="🔍 Search complete")
        
        self.refresh_notes_list()
    
    def clear_search(self):
        """Clear search and show all notes"""
        self.search_var.set("")
        self.robot_status.configure(text="💭 Ready")
    
    def refresh_notes_list(self):
        """Refresh the notes list display"""
        self.notes_listbox.delete(0, tk.END)
        
        # Sort notes: pinned first, then by update time
        sorted_notes = sorted(
            self.filtered_notes,
            key=lambda n: (not n.pinned, n.updated_at),
            reverse=True
        )
        
        for note in sorted_notes:
            title = note.title or "Untitled"
            pin_indicator = "📌 " if note.pinned else ""
            date_str = datetime.fromisoformat(note.updated_at).strftime("%m/%d %H:%M")
            
            # Add summary preview if available
            summary_preview = ""
            if note.summary:
                summary_preview = f" • {note.summary[:30]}..."
            
            display_text = f"{pin_indicator}{title} ({date_str}){summary_preview}"
            self.notes_listbox.insert(tk.END, display_text)
        
        # Update filtered notes and note count
        self.filtered_notes = sorted_notes
        self.note_count_var.set(f"📝 {len(self.notes)} notes")
    
    def setup_autosave(self):
        """Setup automatic saving every 30 seconds"""
        def autosave():
            if self.selected_note and self.has_unsaved_changes():
                self.save_current_note(show_status=False)
            self.root.after(30000, autosave)  # 30 seconds
        
        self.root.after(30000, autosave)
    
    def load_notes(self):
        """Load notes from JSON file"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.notes = []
                    for note_data in data:
                        # Handle older format without summary field
                        if 'summary' not in note_data:
                            note_data['summary'] = ""
                        if 'is_markdown' not in note_data:
                            note_data['is_markdown'] = False
                        self.notes.append(Note.from_dict(note_data))
                    logger.info(f"Loaded {len(self.notes)} notes")
            else:
                self.create_sample_notes()
            
            # Initialize filtered notes with all notes
            self.filtered_notes = self.notes.copy()
            
        except Exception as e:
            logger.error(f"Failed to load notes: {e}")
            messagebox.showerror("Error", f"Failed to load notes: {e}")
            self.notes = []
            self.filtered_notes = []
    
    def save_notes(self):
        """Save notes to JSON file"""
        try:
            data = [note.to_dict() for note in self.notes]
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save notes: {e}")
            messagebox.showerror("Error", f"Failed to save notes: {e}")
    
    def create_sample_notes(self):
        """Create sample notes for first-time users"""
        sample_notes = [
            Note(
                id=str(uuid.uuid4()),
                title="Welcome to Hrudhi Personal Assistant",
                content="""# Welcome! 🤖

I'm **Hrudhi**, your personal AI assistant. Here's what I can do for you:

## ✨ AI-Powered Features
- **Smart Search**: Find notes by meaning, not just keywords
- **AI Summarization**: Get instant summaries of your notes
- **Markdown Support**: Rich text formatting with live preview

## 🎨 Modern Design
- **Glassmorphism UI**: Beautiful, modern interface
- **Dark/Light Themes**: Automatically adapts to your system
- **3D Robot Avatar**: I'm always here to assist you!

## 📝 Note Management
- Pin important notes with 📌
- Auto-save every 30 seconds
- Tags with #hashtags
- Import/export functionality

Try typing `#important` or `#work` to see how tags work!

**Pro tip**: Use the AI Summary button to get quick overviews of long notes.""",
                summary="Welcome guide covering AI features, modern design, and note management capabilities.",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=["welcome", "guide", "important"],
                pinned=True,
                is_markdown=True
            ),
            Note(
                id=str(uuid.uuid4()),
                title="Markdown Example",
                content="""# Markdown Formatting Examples

## Headers
Use # for headers (H1), ## for H2, etc.

## Text Formatting
- **Bold text** with double asterisks
- *Italic text* with single asterisks
- `Inline code` with backticks

## Links and Lists
- [Visit GitHub](https://github.com)
- Create bullet points with dashes
- 1. Or numbered lists
- 2. Like this

## Code Blocks
```python
def hello_world():
    print("Hello from Hrudhi!")
```

Enable Markdown mode in the toolbar to see live formatting! #markdown #example""",
                summary="Examples of markdown formatting including headers, text styles, lists, and code blocks.",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=["markdown", "example", "formatting"]
            )
        ]
        
        self.notes = sample_notes
        self.filtered_notes = self.notes.copy()  # Initialize filtered notes
        self.save_notes()
    
    def import_notes(self):
        """Import notes from JSON file"""
        file_path = filedialog.askopenfilename(
            title="Import Notes",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    imported_notes = []
                    for note_data in data:
                        # Handle older format
                        if 'summary' not in note_data:
                            note_data['summary'] = ""
                        if 'is_markdown' not in note_data:
                            note_data['is_markdown'] = False
                        imported_notes.append(Note.from_dict(note_data))
                    
                    self.notes.extend(imported_notes)
                    self.save_notes()
                    self.refresh_notes_list()
                    self.status_var.set(f"Imported {len(imported_notes)} notes successfully! 📁✨")
                    
                    # Update robot mood
                    if self.robot:
                        self.robot.set_mood("happy")
                        
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import notes: {e}")
    
    def export_notes(self):
        """Export notes to JSON file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Notes",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = [note.to_dict() for note in self.notes]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.status_var.set(f"Exported {len(self.notes)} notes successfully! 📤✨")
                
                # Update robot mood
                if self.robot:
                    self.robot.set_mood("happy")
                    
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export notes: {e}")
    
    def open_chat_assistant(self):
        """Open the AI chat assistant"""
        try:
            import subprocess
            import sys
            
            # Check if we have notes to chat about
            if not self.notes:
                result = messagebox.askyesno(
                    "No Notes Found", 
                    "You don't have any notes saved yet. The chat assistant works best when you have some notes to discuss.\n\nWould you like to continue anyway?"
                )
                if not result:
                    return
            
            # Show info message
            messagebox.showinfo(
                "Starting Chat Assistant",
                "🤖 Opening Hrudhi Chat Assistant!\n\n" +
                "• First time may take 1-2 minutes to download AI models\n" +
                "• The assistant knows about all your saved notes\n" +
                "• Ask questions about your content!\n\n" +
                "A new window will open shortly..."
            )
            
            # Update status and robot mood
            self.status_var.set("🤖 Starting AI Chat Assistant...")
            if self.robot:
                self.robot.set_mood("thinking")
            
            # Launch chat assistant
            script_path = Path(__file__).parent / "hrudhi_chat_assistant.py"
            subprocess.Popen([sys.executable, str(script_path)])
            
            self.status_var.set("🤖 Chat Assistant opened! Check the new window.")
            
        except Exception as e:
            messagebox.showerror("Chat Error", f"Failed to open chat assistant: {e}")
            self.status_var.set("❌ Failed to open chat assistant")
    
    def run(self):
        """Start the application"""
        logger.info("Starting Hrudhi Personal Assistant")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.selected_note and self.has_unsaved_changes():
            self.save_current_note()
        
        logger.info("Hrudhi Personal Assistant closing")
        
        # Clean up pygame
        try:
            pygame.quit()
        except:
            pass
            
        self.root.destroy()

def main():
    """Main entry point"""
    try:
        app = HrudhiPersonalAssistant()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        messagebox.showerror("Error", f"Hrudhi Personal Assistant failed to start: {e}")

if __name__ == "__main__":
    main()