import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import random

# Configuration
NOTES_DIR = os.path.expanduser("~/Desktop/HrudhiNotes")
EMBEDDINGS_FILE = os.path.join(NOTES_DIR, "embeddings.json")

# Initialize AI model and storage
model = None
embeddings_db = {}

def initialize():
    global model, embeddings_db
    os.makedirs(NOTES_DIR, exist_ok=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
            embeddings_db = {k: v for k, v in embeddings_data.items()}

def save_note(text, topic):
    """Save note with AI embedding"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{topic.replace(' ', '_')[:20]}.txt"
    filepath = os.path.join(NOTES_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Generate and save embedding
    embedding = model.encode(text).tolist()
    embeddings_db[filename] = embedding
    
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(embeddings_db, f, ensure_ascii=False, indent=2)
    
    return filename

def search_notes(query, top_k=5, min_similarity=0.25):
    """Advanced context-aware search with memory of all typed content"""
    query_embedding = model.encode(query).reshape(1, -1)
    results = []
    
    for filename, embedding in embeddings_db.items():
        similarity = cosine_similarity(query_embedding, [embedding])[0][0]
        
        # Lower threshold for better context matching
        if similarity >= min_similarity:
            # Read file content for preview and context analysis
            filepath = os.path.join(NOTES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Enhanced context matching - find similar phrases/concepts
                    query_words = set(query.lower().split())
                    content_words = set(content.lower().split())
                    word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
                    
                    # Combined score: AI similarity + word overlap
                    combined_score = (similarity * 0.7) + (word_overlap * 0.3)
                    
                    # Create smart preview highlighting relevant parts
                    preview = create_smart_preview(content, query, 120)
                    
                    # Extract metadata from filename
                    timestamp_part = filename[:15] if filename[8] == '_' else filename[:8]
                    category = filename[16:].replace('.txt', '') if len(filename) > 16 else "general"
                    
                    results.append({
                        'filename': filename,
                        'similarity': similarity,
                        'combined_score': combined_score,
                        'preview': preview,
                        'content': content,
                        'category': category,
                        'timestamp': timestamp_part,
                        'word_match': word_overlap
                    })
            except Exception:
                results.append({
                    'filename': filename,
                    'similarity': similarity,
                    'combined_score': similarity,
                    'preview': "Preview unavailable",
                    'content': "",
                    'category': "unknown",
                    'timestamp': "",
                    'word_match': 0
                })
    
    # Sort by combined score for better context relevance
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results[:top_k]

def create_smart_preview(content, query, max_length=120):
    """Create intelligent preview highlighting query-relevant parts"""
    query_words = query.lower().split()
    content_lower = content.lower()
    
    # Find the best section that contains query words
    best_start = 0
    best_score = 0
    
    # Sliding window to find most relevant section
    for start in range(0, len(content), 20):
        end = min(start + max_length, len(content))
        section = content_lower[start:end]
        
        # Count query word matches in this section
        score = sum(word in section for word in query_words)
        if score > best_score:
            best_score = score
            best_start = start
    
    # Extract the best section
    end = min(best_start + max_length, len(content))
    preview = content[best_start:end].strip()
    
    if best_start > 0:
        preview = "..." + preview
    if end < len(content):
        preview = preview + "..."
        
    return preview

class RobotFace:
    def __init__(self, canvas, x, y, size=60):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.size = size
        self.mood = "neutral"
        self.blink_counter = 0
        self.bounce_offset = 0
        self.create_cute_3d_robot()
        self.animate()
    
    def create_cute_3d_robot(self):
        # Create adorable 3D-style robot like the image
        
        # Main body (rounded cylinder with gradient effect)
        self.body = self.canvas.create_oval(
            self.x - self.size//2, self.y - self.size//3,
            self.x + self.size//2, self.y + self.size//2,
            fill="#F8F9FA", outline="#E9ECEF", width=2
        )
        
        # Blue accent stripe (like the image)
        self.stripe = self.canvas.create_arc(
            self.x - self.size//2 + 5, self.y - self.size//6,
            self.x + self.size//2 - 5, self.y + self.size//4,
            start=0, extent=180, fill="#4FC3F7", outline="#29B6F6", width=1
        )
        
        # Head/helmet area (rounded top)
        self.head = self.canvas.create_arc(
            self.x - self.size//2.2, self.y - self.size//2,
            self.x + self.size//2.2, self.y + self.size//8,
            start=0, extent=180, fill="#F8F9FA", outline="#E9ECEF", width=2
        )
        
        # Cute screen face (like the image)
        self.screen = self.canvas.create_rounded_rect(
            self.x - self.size//3, self.y - self.size//3,
            self.x + self.size//3, self.y - self.size//8,
            radius=8, fill="#263238", outline="#37474F", width=1
        )
        
        # Adorable glowing eyes (cyan like the image)
        eye_size = 4
        eye_glow_size = 6
        eye_y = self.y - self.size//4.5
        
        # Eye glow effect
        self.left_eye_glow = self.canvas.create_oval(
            self.x - self.size//6 - eye_glow_size//2, eye_y - eye_glow_size//2,
            self.x - self.size//6 + eye_glow_size//2, eye_y + eye_glow_size//2,
            fill="#4DD0E1", outline="", width=0
        )
        self.right_eye_glow = self.canvas.create_oval(
            self.x + self.size//6 - eye_glow_size//2, eye_y - eye_glow_size//2,
            self.x + self.size//6 + eye_glow_size//2, eye_y + eye_glow_size//2,
            fill="#4DD0E1", outline="", width=0
        )
        
        # Bright cyan eyes
        self.left_eye = self.canvas.create_oval(
            self.x - self.size//6 - eye_size//2, eye_y - eye_size//2,
            self.x - self.size//6 + eye_size//2, eye_y + eye_size//2,
            fill="#00E5FF", outline="#00BCD4", width=1
        )
        self.right_eye = self.canvas.create_oval(
            self.x + self.size//6 - eye_size//2, eye_y - eye_size//2,
            self.x + self.size//6 + eye_size//2, eye_y + eye_size//2,
            fill="#00E5FF", outline="#00BCD4", width=1
        )
        
        # Cute smile (small arc)
        self.mouth = self.canvas.create_arc(
            self.x - self.size//8, self.y - self.size//7,
            self.x + self.size//8, self.y - self.size//12,
            start=0, extent=180, outline="#4DD0E1", width=2, style="arc"
        )
        
        # Cute little antennas/sensors (like headphones in the image)
        antenna_y = self.y - self.size//2.2
        self.left_antenna = self.canvas.create_oval(
            self.x - self.size//2.5, antenna_y - 3,
            self.x - self.size//2.5 + 6, antenna_y + 3,
            fill="#4FC3F7", outline="#29B6F6", width=1
        )
        self.right_antenna = self.canvas.create_oval(
            self.x + self.size//2.5 - 6, antenna_y - 3,
            self.x + self.size//2.5, antenna_y + 3,
            fill="#4FC3F7", outline="#29B6F6", width=1
        )
        
        # Little connection lines
        self.left_line = self.canvas.create_line(
            self.x - self.size//2.5 + 3, antenna_y,
            self.x - self.size//3, self.y - self.size//3,
            fill="#B0BEC5", width=1
        )
        self.right_line = self.canvas.create_line(
            self.x + self.size//2.5 - 3, antenna_y,
            self.x + self.size//3, self.y - self.size//3,
            fill="#B0BEC5", width=1
        )
    
    def animate(self):
        # Gentle floating/breathing animation like the cute robot
        self.bounce_offset = (self.bounce_offset + 1) % 60
        bounce = 2 * (0.5 + 0.5 * (self.bounce_offset / 30.0 - 1) ** 2)
        
        # Move all components slightly up and down
        for item in [self.body, self.stripe, self.head, self.screen, 
                    self.left_eye, self.right_eye, self.left_eye_glow, self.right_eye_glow,
                    self.mouth, self.left_antenna, self.right_antenna, 
                    self.left_line, self.right_line]:
            self.canvas.coords(item, *[coord + (bounce - 1) if i % 2 == 1 else coord 
                                     for i, coord in enumerate(self.canvas.coords(item))])
        
        # Blinking animation for cuteness
        if self.bounce_offset % 40 == 0:  # Blink every 40 frames
            self.blink()
        
        self.canvas.after(50, self.animate)
    
    def blink(self):
        # Cute blinking animation
        original_eye_color = "#00E5FF"
        self.canvas.itemconfig(self.left_eye, fill="#263238")
        self.canvas.itemconfig(self.right_eye, fill="#263238")
        self.canvas.after(100, lambda: [
            self.canvas.itemconfig(self.left_eye, fill=original_eye_color),
            self.canvas.itemconfig(self.right_eye, fill=original_eye_color)
        ])
    
    def set_mood(self, mood):
        self.mood = mood
        if mood == "happy":
            # Brighter eyes when happy
            self.canvas.itemconfig(self.left_eye, fill="#00FFFF")
            self.canvas.itemconfig(self.right_eye, fill="#00FFFF")
        elif mood == "thinking":
            # Dimmer, pulsing eyes when thinking
            self.canvas.itemconfig(self.left_eye, fill="#26C6DA")
            self.canvas.itemconfig(self.right_eye, fill="#26C6DA")
        else:
            # Normal cyan
            self.canvas.itemconfig(self.left_eye, fill="#00E5FF")
            self.canvas.itemconfig(self.right_eye, fill="#00E5FF")

# Add rounded rectangle method to Canvas
def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
    points = []
    for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                 (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                 (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                 (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
        points.extend([x, y])
    return self.create_polygon(points, smooth=True, **kwargs)

tk.Canvas.create_rounded_rect = create_rounded_rect

class ModernCard(tk.Frame):
    """Modern card component for search results"""
    def __init__(self, parent, title, content, similarity, **kwargs):
        super().__init__(parent, bg='white', relief='flat', bd=1, **kwargs)
        self.configure(highlightbackground='#E5E7EB', highlightthickness=1)
        
        # Title
        title_label = tk.Label(self, text=title, bg='white', fg='#1F2937',
                              font=('Segoe UI', 11, 'bold'), anchor='w')
        title_label.pack(fill='x', padx=16, pady=(12, 4))
        
        # Content preview
        content_label = tk.Label(self, text=content, bg='white', fg='#6B7280',
                               font=('Segoe UI', 9), anchor='w', wraplength=400, justify='left')
        content_label.pack(fill='x', padx=16, pady=(0, 8))
        
        # Relevance score
        score_frame = tk.Frame(self, bg='white')
        score_frame.pack(fill='x', padx=16, pady=(0, 12))
        
        relevance_label = tk.Label(score_frame, text=f"Relevance: {similarity:.0%}", 
                                 bg='white', fg='#6366F1', font=('Segoe UI', 8, 'bold'))
        relevance_label.pack(side='left')
        
        # Score bar
        score_bar = tk.Frame(score_frame, height=4, bg='#E5E7EB')
        score_bar.pack(side='right', fill='x', expand=True, padx=(8, 0))
        
        filled_width = int(similarity * 100)
        if filled_width > 0:
            fill_bar = tk.Frame(score_bar, height=4, bg='#6366F1')
            fill_bar.place(x=0, y=0, width=f"{filled_width}%", height=4)

class HrudhiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hrudhi - Your Adorable AI Companion 🤖")
        self.root.geometry("900x700")
        self.root.configure(bg='#F0F8FF')
        
        self.create_styles()
        self.create_widgets()
        self.show_welcome_message()
        
        # Load existing notes into memory display
        self.load_recent_notes()
    
    def create_styles(self):
        # Modern color palette
        self.colors = {
            'primary': '#6366F1',      # Modern indigo
            'secondary': '#F59E0B',    # Warm amber
            'surface': '#FAFAFA',      # Light surface
            'surface_dark': '#F5F5F5', # Slightly darker surface
            'text_primary': '#1F2937',  # Dark text
            'text_secondary': '#6B7280', # Gray text
            'accent': '#10B981',       # Success green
            'border': '#E5E7EB'        # Light border
        }
        
        # Configure modern ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Modern.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       focuscolor='none',
                       padding=(16, 8))
        
        style.map('Modern.TButton',
                 background=[('active', '#5B5FED'),
                           ('pressed', '#4F46E5')])
        
        style.configure('Secondary.TButton',
                       background=self.colors['secondary'],
                       foreground='white',
                       font=('Segoe UI', 10),
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 6))
        
        style.configure('Modern.TNotebook', 
                       background=self.colors['surface'],
                       borderwidth=0)
        
        style.configure('Modern.TNotebook.Tab',
                       background=self.colors['surface_dark'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10),
                       padding=(16, 8))
        
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', 'white')])
    
    def create_widgets(self):
        # Modern main container with cute robot theme
        self.root.configure(bg='#F0F8FF')  # Alice blue background
        main_frame = tk.Frame(self.root, bg='#F0F8FF')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        
        # Adorable header with cute robot
        header_frame = tk.Frame(main_frame, bg='#F0F8FF')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Cute robot (bigger for more presence)
        self.robot_canvas = tk.Canvas(header_frame, width=80, height=80, 
                                    bg='#F0F8FF', highlightthickness=0)
        self.robot_canvas.pack(side=tk.LEFT, padx=(0, 16))
        
        # Initialize adorable robot
        self.robot = RobotFace(self.robot_canvas, 40, 40, 70)
        
        # Friendly title section
        title_frame = tk.Frame(header_frame, bg='#F0F8FF')
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(title_frame, text="Hi! I'm Hrudhi 🤖", 
                             bg='#F0F8FF', fg='#1565C0',
                             font=('Segoe UI', 18, 'bold'))
        title_label.pack(anchor=tk.W)
        
        subtitle_label = tk.Label(title_frame, text="Your adorable AI companion for smart notes",
                                bg='#F0F8FF', fg='#424242',
                                font=('Segoe UI', 10))
        subtitle_label.pack(anchor=tk.W)
        
        # Status with cute messages
        self.status_var = tk.StringVar(value="Ready to help with your brilliant ideas! ✨")
        status_label = tk.Label(title_frame, textvariable=self.status_var,
                              bg='#F0F8FF', fg='#666666',
                              font=('Segoe UI', 9))
        status_label.pack(anchor=tk.W, pady=(4, 0))
        
        # Chat management buttons
        button_frame = tk.Frame(header_frame, bg='#F0F8FF')
        button_frame.pack(side=tk.RIGHT, padx=(16, 0))
        
        # New Chat button
        new_chat_btn = tk.Button(button_frame, text="🆕 New Chat", 
                               command=self.new_chat,
                               bg='#4FC3F7', fg='white', font=('Segoe UI', 9, 'bold'),
                               relief='flat', padx=12, pady=6, cursor='hand2')
        new_chat_btn.pack(pady=2)
        
        # Quick Actions button  
        actions_btn = tk.Button(button_frame, text="⚡ Quick Actions",
                              command=self.show_quick_actions,
                              bg='#26A69A', fg='white', font=('Segoe UI', 9, 'bold'),
                              relief='flat', padx=12, pady=6, cursor='hand2')
        actions_btn.pack(pady=2)
        
        # Modern notebook tabs with chat management
        self.notebook = ttk.Notebook(main_frame, style='Modern.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced tabs
        self.create_enhanced_add_tab()
        self.create_enhanced_search_tab()
        self.create_chat_history_tab()
    
    def create_enhanced_add_tab(self):
        add_frame = tk.Frame(self.notebook, bg='white', padx=20, pady=16)
        self.notebook.add(add_frame, text="✏️ New Note")
        
        # Smart input with context awareness
        input_label = tk.Label(add_frame, text="💭 What's on your mind?", 
                             bg='white', fg=self.colors['text_primary'],
                             font=('Segoe UI', 12, 'bold'))
        input_label.pack(anchor='w', pady=(0, 8))
        
        # Enhanced text area with placeholder
        self.text = tk.Text(add_frame, height=12, wrap=tk.WORD,
                           bg='#FAFAFA', fg=self.colors['text_primary'], 
                           insertbackground=self.colors['primary'],
                           font=('Segoe UI', 11), padx=16, pady=16,
                           relief=tk.FLAT, bd=1,
                           selectbackground='#E3F2FD',
                           selectforeground=self.colors['text_primary'],
                           highlightbackground='#E1F5FE',
                           highlightthickness=2)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # Smart category with suggestions
        category_frame = tk.Frame(add_frame, bg='white')
        category_frame.pack(fill=tk.X, pady=(0, 12))
        
        category_label = tk.Label(category_frame, text="🏷️ Category:", 
                                bg='white', fg=self.colors['text_secondary'],
                                font=('Segoe UI', 10))
        category_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.topic_entry = tk.Entry(category_frame, bg='#FAFAFA', 
                                  fg=self.colors['text_primary'],
                                  insertbackground=self.colors['primary'], 
                                  font=('Segoe UI', 10), relief=tk.FLAT, bd=1,
                                  highlightbackground='#E1F5FE', highlightthickness=1)
        self.topic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        # Category suggestions
        suggestions = ["💡 ideas", "📝 meeting", "📚 learning", "✅ tasks", "💭 thoughts"]
        for suggestion in suggestions:
            btn = tk.Button(category_frame, text=suggestion, 
                          command=lambda s=suggestion: self.set_category(s),
                          bg='#E8F5E8', fg='#2E7D32', font=('Segoe UI', 8),
                          relief='flat', padx=6, pady=2, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=2)
        
        # Enhanced save button
        save_frame = tk.Frame(add_frame, bg='white')
        save_frame.pack(fill=tk.X)
        
        save_btn = tk.Button(save_frame, text="💾 Save & Remember", 
                           command=self.save_note,
                           bg='#1976D2', fg='white', font=('Segoe UI', 11, 'bold'),
                           relief='flat', padx=20, pady=10, cursor='hand2')
        save_btn.pack(side=tk.RIGHT)
    
    def create_enhanced_search_tab(self):
        search_frame = tk.Frame(self.notebook, bg='white', padx=20, pady=16)
        self.notebook.add(search_frame, text="🔍 Smart Search")
        
        # Smart search header
        search_label = tk.Label(search_frame, text="🧠 Context-Aware Search", 
                              bg='white', fg=self.colors['text_primary'],
                              font=('Segoe UI', 12, 'bold'))
        search_label.pack(anchor='w', pady=(0, 8))
        
        help_label = tk.Label(search_frame, text="I remember everything you've typed and can find similar contexts!", 
                            bg='white', fg=self.colors['text_secondary'],
                            font=('Segoe UI', 9))
        help_label.pack(anchor='w', pady=(0, 12))
        
        # Enhanced search input
        search_input_frame = tk.Frame(search_frame, bg='white')
        search_input_frame.pack(fill=tk.X, pady=(0, 16))
        
        self.search_entry = tk.Entry(search_input_frame, bg='#FAFAFA', 
                                   fg=self.colors['text_primary'],
                                   insertbackground=self.colors['primary'], 
                                   font=('Segoe UI', 12), relief=tk.FLAT, bd=1,
                                   highlightbackground='#E1F5FE', highlightthickness=2)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.search_entry.bind('<Return>', lambda e: self.search_notes())
        
        search_btn = tk.Button(search_input_frame, text="🔍 Find Similar", 
                             command=self.search_notes,
                             bg='#FF7043', fg='white', font=('Segoe UI', 10, 'bold'),
                             relief='flat', padx=16, pady=8, cursor='hand2')
        search_btn.pack(side=tk.RIGHT)
        
        # Results area with better styling
        self.results_frame = tk.Frame(search_frame, bg='white')
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced scrollable results
        self.results_canvas = tk.Canvas(self.results_frame, bg='#FAFAFA', 
                                      highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, 
                                        command=self.results_canvas.yview)
        self.scrollable_results = tk.Frame(self.results_canvas, bg='#FAFAFA')
        
        self.scrollable_results.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_results, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel support
        def _on_mousewheel(event):
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.results_canvas.bind("<MouseWheel>", _on_mousewheel)
    
    def create_chat_history_tab(self):
        history_frame = tk.Frame(self.notebook, bg='white', padx=20, pady=16)
        self.notebook.add(history_frame, text="📚 My Memory")
        
        # Memory overview
        memory_label = tk.Label(history_frame, text="🧠 What I Remember About You", 
                              bg='white', fg=self.colors['text_primary'],
                              font=('Segoe UI', 12, 'bold'))
        memory_label.pack(anchor='w', pady=(0, 16))
        
        # Stats frame
        stats_frame = tk.Frame(history_frame, bg='#E8F5E8', relief='flat', bd=1)
        stats_frame.pack(fill=tk.X, pady=(0, 16))
        
        # Note count and categories
        note_count = len(embeddings_db)
        stats_text = f"📊 Total Notes: {note_count} | 🏷️ Categories discovered | 🕒 Last activity"
        
        stats_label = tk.Label(stats_frame, text=stats_text,
                             bg='#E8F5E8', fg='#2E7D32',
                             font=('Segoe UI', 10, 'bold'), pady=12)
        stats_label.pack()
        
        # Recent activity list
        recent_frame = tk.Frame(history_frame, bg='white')
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        recent_label = tk.Label(recent_frame, text="📝 Recent Notes", 
                              bg='white', fg=self.colors['text_primary'],
                              font=('Segoe UI', 11, 'bold'))
        recent_label.pack(anchor='w', pady=(0, 8))
        
        # Recent notes listbox
        self.recent_listbox = tk.Listbox(recent_frame, bg='#FAFAFA',
                                       font=('Segoe UI', 10), height=15,
                                       selectbackground='#E3F2FD',
                                       relief='flat', bd=1)
        self.recent_listbox.pack(fill=tk.BOTH, expand=True)
        self.recent_listbox.bind('<Double-1>', self.open_from_history)
        
        # Load recent notes
        self.load_recent_notes()
    
    def create_add_note_tab(self):
        add_frame = tk.Frame(self.notebook, bg='white', padx=24, pady=20)
        self.notebook.add(add_frame, text="✏️ Add Note")
        
        # Note input
        note_label = tk.Label(add_frame, text="What's on your mind?", 
                            bg='white', fg=self.colors['text_primary'],
                            font=('Segoe UI', 12, 'bold'))
        note_label.pack(anchor='w', pady=(0, 8))
        
        self.text = tk.Text(add_frame, height=12, wrap=tk.WORD,
                           bg='white', fg=self.colors['text_primary'], 
                           insertbackground=self.colors['primary'],
                           font=('Segoe UI', 11), padx=16, pady=16,
                           relief=tk.FLAT, bd=1,
                           selectbackground=self.colors['primary'],
                           selectforeground='white',
                           highlightbackground=self.colors['border'],
                           highlightthickness=1)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 16))
        
        # Category input
        category_frame = tk.Frame(add_frame, bg='white')
        category_frame.pack(fill=tk.X, pady=(0, 20))
        
        category_label = tk.Label(category_frame, text="Category:", 
                                bg='white', fg=self.colors['text_secondary'],
                                font=('Segoe UI', 10))
        category_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.topic_entry = tk.Entry(category_frame, bg='white', 
                                  fg=self.colors['text_primary'],
                                  insertbackground=self.colors['primary'], 
                                  font=('Segoe UI', 10),
                                  relief=tk.FLAT, bd=1,
                                  highlightbackground=self.colors['border'],
                                  highlightthickness=1)
        self.topic_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Save button
        save_btn = ttk.Button(add_frame, text="💾 Save Note", 
                            command=self.save_note, style='Modern.TButton')
        save_btn.pack(anchor=tk.E, pady=(8, 0))
    
    def create_search_tab(self):
        search_frame = tk.Frame(self.notebook, bg='white', padx=24, pady=20)
        self.notebook.add(search_frame, text="🔍 Search")
        
        # Search input
        search_label = tk.Label(search_frame, text="Search your notes:", 
                              bg='white', fg=self.colors['text_primary'],
                              font=('Segoe UI', 12, 'bold'))
        search_label.pack(anchor='w', pady=(0, 8))
        
        search_input_frame = tk.Frame(search_frame, bg='white')
        search_input_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.search_entry = tk.Entry(search_input_frame, bg='white', 
                                   fg=self.colors['text_primary'],
                                   insertbackground=self.colors['primary'], 
                                   font=('Segoe UI', 11),
                                   relief=tk.FLAT, bd=1,
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.search_entry.bind('<Return>', lambda e: self.search_notes())
        
        search_btn = ttk.Button(search_input_frame, text="Search", 
                              command=self.search_notes, style='Secondary.TButton')
        search_btn.pack(side=tk.RIGHT)
        
        # Results area
        self.results_frame = tk.Frame(search_frame, bg='white')
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable results
        self.results_canvas = tk.Canvas(self.results_frame, bg='white', 
                                      highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, 
                                        command=self.results_canvas.yview)
        self.scrollable_results = tk.Frame(self.results_canvas, bg='white')
        
        self.scrollable_results.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_results, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel binding
        def _on_mousewheel(event):
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.results_canvas.bind("<MouseWheel>", _on_mousewheel)
    
    def show_welcome_message(self):
        messages = [
            "Ready to capture your brilliant ideas! 💡✨",
            "What amazing thoughts shall we save today? 🤖💭", 
            "I'm here to remember everything important! 🧠💾",
            "Let's create something wonderful together! 🌟📝"
        ]
        self.status_var.set(random.choice(messages))
        self.robot.set_mood("happy")
    
    def new_chat(self):
        """Start a fresh conversation"""
        self.text.delete("1.0", tk.END)
        self.topic_entry.delete(0, tk.END)
        self.search_entry.delete(0, tk.END)
        
        # Clear search results
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()
            
        self.status_var.set("🆕 Fresh start! What new ideas do you have?")
        self.robot.set_mood("excited")
        self.notebook.select(0)  # Switch to add note tab
    
    def show_quick_actions(self):
        """Show quick action popup"""
        actions_window = tk.Toplevel(self.root)
        actions_window.title("Quick Actions")
        actions_window.geometry("300x400")
        actions_window.configure(bg='white')
        actions_window.transient(self.root)
        actions_window.grab_set()
        
        # Center the window
        actions_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        title = tk.Label(actions_window, text="⚡ Quick Actions", 
                        bg='white', fg='#1565C0',
                        font=('Segoe UI', 14, 'bold'))
        title.pack(pady=16)
        
        # Action buttons
        actions = [
            ("🔍 Find Similar Notes", lambda: self.quick_search(actions_window)),
            ("📝 Continue Last Note", lambda: self.continue_last_note(actions_window)),
            ("🏷️ Browse by Category", lambda: self.browse_categories(actions_window)),
            ("📊 Show Statistics", lambda: self.show_stats(actions_window)),
            ("🗑️ Clean Up Old Notes", lambda: self.cleanup_notes(actions_window)),
        ]
        
        for text, command in actions:
            btn = tk.Button(actions_window, text=text, command=command,
                          bg='#E3F2FD', fg='#1565C0', font=('Segoe UI', 10),
                          relief='flat', padx=20, pady=8, cursor='hand2')
            btn.pack(fill='x', padx=20, pady=4)
    
    def set_category(self, category):
        """Set category from suggestion button"""
        self.topic_entry.delete(0, tk.END)
        self.topic_entry.insert(0, category)
    
    def load_recent_notes(self):
        """Load recent notes in history tab"""
        self.recent_listbox.delete(0, tk.END)
        
        # Get all notes sorted by filename (which contains timestamp)
        notes = list(embeddings_db.keys())
        notes.sort(reverse=True)  # Most recent first
        
        for i, note in enumerate(notes[:20]):  # Show last 20 notes
            # Clean up display name
            display_name = note.replace('.txt', '').replace('_', ' ')
            if len(display_name) > 8 and display_name[8] == ' ':
                timestamp_part = display_name[:8]
                title_part = display_name[9:]
                display = f"📝 {title_part} ({timestamp_part})"
            else:
                display = f"📝 {display_name}"
            
            self.recent_listbox.insert(tk.END, display)
    
    def open_from_history(self, event):
        """Open note from history list"""
        selection = self.recent_listbox.curselection()
        if selection:
            selected_index = selection[0]
            notes = list(embeddings_db.keys())
            notes.sort(reverse=True)
            
            if selected_index < len(notes):
                filename = notes[selected_index]
                filepath = os.path.join(NOTES_DIR, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.show_note_for_editing(filename, content)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open note: {e}")
    
    def show_note_for_editing(self, filename, content):
        """Show note in editing window with modify options"""
        edit_window = tk.Toplevel(self.root)
        edit_window.title(f"Edit: {filename.replace('.txt', '').replace('_', ' ')}")
        edit_window.geometry("700x500")
        edit_window.configure(bg='white')
        
        # Header
        header_frame = tk.Frame(edit_window, bg='#E3F2FD')
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(header_frame, text=f"✏️ Editing: {filename.replace('.txt', '').replace('_', ' ')}", 
                             bg='#E3F2FD', fg='#1565C0',
                             font=('Segoe UI', 12, 'bold'))
        title_label.pack(pady=10)
        
        # Content editor
        text_editor = tk.Text(edit_window, wrap=tk.WORD, bg='#FAFAFA',
                            fg='#1F2937', font=('Segoe UI', 11),
                            padx=20, pady=20)
        text_editor.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        text_editor.insert(tk.END, content)
        
        # Buttons
        button_frame = tk.Frame(edit_window, bg='white')
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        def save_changes():
            new_content = text_editor.get("1.0", tk.END).strip()
            if new_content:
                # Save updated content
                filepath = os.path.join(NOTES_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                # Update embedding
                embedding = model.encode(new_content).tolist()
                embeddings_db[filename] = embedding
                
                with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_db, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Success", "Note updated successfully!")
                edit_window.destroy()
                self.load_recent_notes()  # Refresh history
        
        def create_new_version():
            new_content = text_editor.get("1.0", tk.END).strip()
            if new_content:
                # Create new note with current timestamp
                category = filename.split('_')[1].replace('.txt', '') if '_' in filename else 'modified'
                save_note(new_content, category)
                messagebox.showinfo("Success", "New version created!")
                edit_window.destroy()
        
        save_btn = tk.Button(button_frame, text="💾 Save Changes", command=save_changes,
                           bg='#4CAF50', fg='white', font=('Segoe UI', 10, 'bold'),
                           relief='flat', padx=16, pady=8, cursor='hand2')
        save_btn.pack(side='left', padx=(0, 8))
        
        new_btn = tk.Button(button_frame, text="🆕 Save as New", command=create_new_version,
                          bg='#2196F3', fg='white', font=('Segoe UI', 10, 'bold'),
                          relief='flat', padx=16, pady=8, cursor='hand2')
        new_btn.pack(side='left', padx=8)
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=edit_window.destroy,
                             bg='#757575', fg='white', font=('Segoe UI', 10, 'bold'),
                             relief='flat', padx=16, pady=8, cursor='hand2')
        cancel_btn.pack(side='right')
    
    def save_note(self):
        text = self.text.get("1.0", tk.END).strip()
        topic = self.topic_entry.get().strip()
        
        if not text:
            messagebox.showwarning("Missing Content", "Please enter some content for your note.")
            return
        
        if not topic:
            topic = "general"
        
        self.status_var.set("💾 Saving and learning from your note...")
        self.robot.set_mood("thinking")
        
        def save_async():
            try:
                filename = save_note(text, topic)
                self.root.after(0, lambda: self.save_complete(filename))
            except Exception as e:
                self.root.after(0, lambda: self.save_error(str(e)))
        
        threading.Thread(target=save_async, daemon=True).start()
    
    def save_complete(self, filename):
        self.text.delete("1.0", tk.END)
        self.topic_entry.delete(0, tk.END)
        self.status_var.set("✅ Note saved successfully!")
        self.robot.set_mood("happy")
        
        # Reset status after 3 seconds
        self.root.after(3000, lambda: self.status_var.set("Ready for your next brilliant idea!"))
    
    def save_error(self, error):
        self.status_var.set(f"❌ Error saving note: {error}")
        messagebox.showerror("Save Error", f"Could not save note: {error}")
    
    def search_notes(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Missing Query", "Please enter a search term.")
            return
        
        self.status_var.set("🔍 Searching through your notes...")
        self.robot.set_mood("thinking")
        
        # Clear previous results
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()
        
        def search_async():
            try:
                results = search_notes(query)
                self.root.after(0, lambda: self.search_complete(results, query))
            except Exception as e:
                self.root.after(0, lambda: self.search_error(str(e)))
        
        threading.Thread(target=search_async, daemon=True).start()
    
    def search_complete(self, results, query):
        # Clear previous results
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()
        
        if not results:
            no_results = tk.Label(self.scrollable_results, 
                                text="🤔 No matching context found. Try different keywords or phrases!",
                                bg='#FAFAFA', fg=self.colors['text_secondary'],
                                font=('Segoe UI', 11), pady=40)
            no_results.pack()
            self.status_var.set("No context matches found. Try different search terms.")
            self.robot.set_mood("thinking")
        else:
            self.status_var.set(f"🎯 Found {len(results)} contextually similar notes!")
            self.robot.set_mood("excited")
            
            for i, result in enumerate(results):
                # Create enhanced result card
                card_frame = tk.Frame(self.scrollable_results, bg='white', relief='flat', bd=1)
                card_frame.configure(highlightbackground='#E0E0E0', highlightthickness=1)
                card_frame.pack(fill='x', pady=8, padx=8)
                
                # Title with category and timestamp
                title_frame = tk.Frame(card_frame, bg='white')
                title_frame.pack(fill='x', padx=16, pady=(12, 4))
                
                title_text = f"📄 {result['category'].replace('_', ' ').title()}"
                title_label = tk.Label(title_frame, text=title_text, bg='white', 
                                     fg='#1565C0', font=('Segoe UI', 11, 'bold'), anchor='w')
                title_label.pack(side='left')
                
                time_label = tk.Label(title_frame, text=f"🕒 {result['timestamp']}", 
                                    bg='white', fg='#757575', font=('Segoe UI', 8), anchor='e')
                time_label.pack(side='right')
                
                # Smart preview with highlighted context
                preview_label = tk.Label(card_frame, text=result['preview'], bg='white', 
                                       fg='#424242', font=('Segoe UI', 9),
                                       anchor='w', wraplength=500, justify='left')
                preview_label.pack(fill='x', padx=16, pady=(0, 8))
                
                # Relevance indicators
                relevance_frame = tk.Frame(card_frame, bg='white')
                relevance_frame.pack(fill='x', padx=16, pady=(0, 12))
                
                # AI similarity score
                ai_score = tk.Label(relevance_frame, text=f"🧠 AI Match: {result['similarity']:.0%}", 
                                  bg='white', fg='#6366F1', font=('Segoe UI', 8, 'bold'))
                ai_score.pack(side='left')
                
                # Word overlap score  
                word_score = tk.Label(relevance_frame, text=f"� Word Match: {result['word_match']:.0%}", 
                                    bg='white', fg='#10B981', font=('Segoe UI', 8, 'bold'))
                word_score.pack(side='left', padx=(12, 0))
                
                # Combined relevance bar
                combined_score = result['combined_score']
                relevance_bar = tk.Frame(relevance_frame, height=4, bg='#E5E7EB', width=100)
                relevance_bar.pack(side='right', padx=(12, 0))
                
                if combined_score > 0:
                    fill_width = max(1, int(combined_score * 100))  # At least 1 pixel
                    fill_color = '#6366F1' if combined_score > 0.7 else '#F59E0B' if combined_score > 0.4 else '#EF4444'
                    fill_bar = tk.Frame(relevance_bar, height=4, bg=fill_color)
                    fill_bar.place(x=0, y=0, width=fill_width, height=4)
                
                # Action buttons
                button_frame = tk.Frame(card_frame, bg='#F8F9FA')
                button_frame.pack(fill='x', padx=8, pady=8)
                
                def view_note(filename=result['filename'], content=result['content']):
                    self.show_note_detail(filename, content)
                
                def edit_note(filename=result['filename'], content=result['content']):
                    self.show_note_for_editing(filename, content)
                
                view_btn = tk.Button(button_frame, text="👁️ View", command=view_note,
                                   bg='#E3F2FD', fg='#1565C0', font=('Segoe UI', 8, 'bold'),
                                   relief='flat', padx=12, pady=4, cursor='hand2')
                view_btn.pack(side='left', padx=4)
                
                edit_btn = tk.Button(button_frame, text="✏️ Edit", command=edit_note,
                                   bg='#FFF3E0', fg='#F57C00', font=('Segoe UI', 8, 'bold'),
                                   relief='flat', padx=12, pady=4, cursor='hand2')
                edit_btn.pack(side='left', padx=4)
                
                # Make entire card clickable to view
                def make_clickable(widget, filename=result['filename'], content=result['content']):
                    widget.bind("<Button-1>", lambda e: self.show_note_detail(filename, content))
                    for child in widget.winfo_children():
                        make_clickable(child, filename, content)
                
                make_clickable(card_frame)
    
    def search_error(self, error):
        self.status_var.set("❌ Search error occurred")
        messagebox.showerror("Search Error", f"Could not search notes: {error}")
    
    def show_note_detail(self, filename, content):
        """Show full note content in a popup"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Note: {filename.replace('.txt', '').replace('_', ' ')}")
        detail_window.geometry("600x400")
        detail_window.configure(bg='white')
        
        # Content display
        text_widget = tk.Text(detail_window, wrap=tk.WORD, bg='white',
                            fg=self.colors['text_primary'], font=('Segoe UI', 11),
                            padx=20, pady=20, state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    # Quick action methods
    def quick_search(self, parent_window):
        parent_window.destroy()
        self.notebook.select(1)  # Switch to search tab
        self.search_entry.focus()
    
    def continue_last_note(self, parent_window):
        parent_window.destroy()
        notes = list(embeddings_db.keys())
        if notes:
            notes.sort(reverse=True)  # Most recent first
            latest_note = notes[0]
            filepath = os.path.join(NOTES_DIR, latest_note)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.show_note_for_editing(latest_note, content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open latest note: {e}")
        else:
            messagebox.showinfo("No Notes", "No notes found to continue!")
    
    def browse_categories(self, parent_window):
        parent_window.destroy()
        categories = {}
        for filename in embeddings_db.keys():
            if '_' in filename:
                category = filename.split('_')[1].replace('.txt', '')
                categories[category] = categories.get(category, 0) + 1
        
        if categories:
            category_window = tk.Toplevel(self.root)
            category_window.title("Browse Categories")
            category_window.geometry("400x300")
            category_window.configure(bg='white')
            
            title = tk.Label(category_window, text="🏷️ Your Note Categories", 
                           bg='white', fg='#1565C0',
                           font=('Segoe UI', 12, 'bold'))
            title.pack(pady=16)
            
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                btn_text = f"{category.replace('_', ' ').title()} ({count} notes)"
                btn = tk.Button(category_window, text=btn_text,
                              command=lambda c=category: self.search_by_category(c, category_window),
                              bg='#E8F5E8', fg='#2E7D32', font=('Segoe UI', 10),
                              relief='flat', padx=20, pady=6, cursor='hand2')
                btn.pack(fill='x', padx=20, pady=2)
    
    def search_by_category(self, category, parent_window):
        parent_window.destroy()
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, category.replace('_', ' '))
        self.notebook.select(1)  # Switch to search tab
        self.search_notes()
    
    def show_stats(self, parent_window):
        parent_window.destroy()
        total_notes = len(embeddings_db)
        
        # Calculate total words
        total_words = 0
        categories = {}
        for filename in embeddings_db.keys():
            filepath = os.path.join(NOTES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_words += len(content.split())
                    
                    if '_' in filename:
                        category = filename.split('_')[1].replace('.txt', '')
                        categories[category] = categories.get(category, 0) + 1
            except:
                pass
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Memory Statistics")
        stats_window.geometry("350x300")
        stats_window.configure(bg='white')
        
        title = tk.Label(stats_window, text="📊 My Memory Stats", 
                        bg='white', fg='#1565C0',
                        font=('Segoe UI', 14, 'bold'))
        title.pack(pady=16)
        
        stats_text = f"""
🧠 Total Notes Remembered: {total_notes}
📝 Total Words Stored: {total_words:,}
🏷️ Categories Created: {len(categories)}
💾 Memory Location: Desktop/HrudhiNotes
        """
        
        stats_label = tk.Label(stats_window, text=stats_text.strip(),
                             bg='white', fg='#424242',
                             font=('Segoe UI', 11), justify='left')
        stats_label.pack(pady=20, padx=20)
        
        if categories:
            cat_label = tk.Label(stats_window, text="📂 Top Categories:",
                               bg='white', fg='#1565C0',
                               font=('Segoe UI', 11, 'bold'))
            cat_label.pack()
            
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                cat_text = f"   • {category.replace('_', ' ').title()}: {count} notes"
                cat_item = tk.Label(stats_window, text=cat_text,
                                  bg='white', fg='#666666',
                                  font=('Segoe UI', 10))
                cat_item.pack()
    
    def cleanup_notes(self, parent_window):
        parent_window.destroy()
        messagebox.showinfo("Cleanup", 
                          "🧹 All your notes are precious memories!\n\n" +
                          "I keep everything organized for you. " +
                          "Use smart search to find what you need!")

def main():
    initialize()
    root = tk.Tk()
    app = HrudhiApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()