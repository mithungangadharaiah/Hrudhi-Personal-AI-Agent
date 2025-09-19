"""
Hrudhi Personal Assistant - Advanced AI-Powered Note Taking & Summarization
Modern Windows 11 design with glassmorphism, 3D avatar, and AI capabilities
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
    """Enhanced AI engine with summarization capabilities"""
    
    def __init__(self):
        self.model = None
        self.summarizer = None
        self.load_models()
    
    def load_models(self):
        """Load AI models for search and summarization"""
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
            
            logger.info("Loading AI models...")
            
            # Load search model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load summarization model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Advanced summarizer failed, using simple model: {e}")
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1
                )
            
            logger.info("✅ AI models loaded successfully")
            
        except ImportError:
            logger.warning("AI libraries not available. AI features disabled.")
            self.model = None
            self.summarizer = None
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            self.model = None
            self.summarizer = None
    
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
        if not self.model or not query.strip():
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
            query_embedding = self.model.encode([query])
            corpus_embeddings = self.model.encode(corpus)
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
        
        # AI and components
        self.ai_engine = AIEngine()
        self.markdown_editor = MarkdownEditor(self)
        
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
        new_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        
        import_btn = ctk.CTkButton(
            controls_frame,
            text="📁 Import",
            command=self.import_notes,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=12)
        )
        import_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        
        # Notes list with enhanced styling
        self.create_enhanced_notes_list()
        
        # Configure sidebar grid
        self.sidebar.grid_rowconfigure(3, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
    
    def create_enhanced_notes_list(self):
        """Create enhanced notes list with modern styling"""
        list_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        list_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
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
        """Create main content area with glassmorphism"""
        main_frame = ctk.CTkFrame(
            self.root,
            corner_radius=20,
            fg_color=("#FFFFFF", "#1E293B"),
            border_width=1,
            border_color=("#E2E8F0", "#334155")
        )
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 15), pady=15)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Note title with enhanced styling
        self.title_entry = ctk.CTkEntry(
            main_frame,
            placeholder_text="✏️ Note title...",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=45,
            corner_radius=15
        )
        self.title_entry.grid(row=0, column=0, sticky="ew", padx=25, pady=(25, 15))
        self.title_entry.bind('<KeyRelease>', self.on_content_changed)
        
        # Enhanced toolbar
        self.create_enhanced_toolbar(main_frame)
        
        # AI Summary section
        self.create_ai_summary_section(main_frame)
        
        # Content editor with markdown support
        self.create_content_editor(main_frame)
    
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
        except Exception as e:
            logger.error(f"Failed to load notes: {e}")
            messagebox.showerror("Error", f"Failed to load notes: {e}")
            self.notes = []
    
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