"""
Hrudhi: Personal AI Note-Taking Agent
Fancy UI with robotic face and creative design
"""
import os
import datetime
import json
import random
import threading
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

NOTES_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "HrudhiNotes")
os.makedirs(NOTES_DIR, exist_ok=True)
EMBEDDINGS_FILE = os.path.join(NOTES_DIR, "embeddings.json")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or initialize embeddings metadata
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings_db = json.load(f)
else:
    embeddings_db = {}

def save_note(text, topic):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{topic}.txt"
    filepath = os.path.join(NOTES_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    embedding = model.encode(text).tolist()
    embeddings_db[filename] = embedding
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings_db, f)
    return filename

def search_notes(query, top_k=5, min_similarity=0.3):
    """Smart search with content previews and relevance filtering"""
    query_embedding = model.encode(query).reshape(1, -1)
    results = []
    
    for filename, embedding in embeddings_db.items():
        similarity = cosine_similarity(query_embedding, [embedding])[0][0]
        
        # Only include results above minimum similarity threshold
        if similarity >= min_similarity:
            # Read file content for preview
            filepath = os.path.join(NOTES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Create a preview (first 100 chars)
                    preview = content[:100] + "..." if len(content) > 100 else content
                    results.append((filename, similarity, preview, content))
            except Exception:
                results.append((filename, similarity, "Preview unavailable", ""))
    
    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# --- Robotic Face and Animation Functions ---
class RobotFace:
    def __init__(self, canvas, x, y, size=80):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.size = size
        self.mood = "neutral"  # neutral, happy, thinking, excited
        self.blink_counter = 0
        self.create_face()
        self.animate()
    
    def create_face(self):
        # Head (circle) - Yellow color like a friendly robot
        self.head = self.canvas.create_oval(
            self.x - self.size//2, self.y - self.size//2,
            self.x + self.size//2, self.y + self.size//2,
            fill="#FFD93D", outline="#FFC107", width=3
        )
        
        # Eyes - Black eyes for cute robot look
        eye_size = self.size // 8
        eye_offset = self.size // 4
        self.left_eye = self.canvas.create_oval(
            self.x - eye_offset - eye_size, self.y - eye_size//2,
            self.x - eye_offset + eye_size, self.y + eye_size//2,
            fill="#000000", outline="#333333", width=2
        )
        self.right_eye = self.canvas.create_oval(
            self.x + eye_offset - eye_size, self.y - eye_size//2,
            self.x + eye_offset + eye_size, self.y + eye_size//2,
            fill="#000000", outline="#333333", width=2
        )
        
        # Mouth (initially neutral smile)
        self.mouth = self.canvas.create_arc(
            self.x - self.size//4, self.y + self.size//8,
            self.x + self.size//4, self.y + self.size//3,
            start=0, extent=180, fill="#FF6B6B", outline="#E55555", width=2
        )
        
        # Antenna with blinking light
        self.canvas.create_line(
            self.x, self.y - self.size//2,
            self.x, self.y - self.size//2 - 20,
            fill="#FFC107", width=3
        )
        self.antenna_light = self.canvas.create_oval(
            self.x - 5, self.y - self.size//2 - 25,
            self.x + 5, self.y - self.size//2 - 15,
            fill="#00FF88", outline="#00CC66", width=2
        )
    
    def set_mood(self, mood):
        self.mood = mood
        if mood == "happy":
            # Happy eyes (black with highlight)
            self.canvas.itemconfig(self.left_eye, fill="#000000")
            self.canvas.itemconfig(self.right_eye, fill="#000000")
            # Happy mouth (big smile)
            self.canvas.coords(
                self.mouth,
                self.x - self.size//3, self.y + self.size//8,
                self.x + self.size//3, self.y + self.size//2
            )
            self.canvas.itemconfig(self.mouth, start=0, extent=180)
        elif mood == "thinking":
            # Thinking eyes (slightly closed)
            self.canvas.itemconfig(self.left_eye, fill="#333333")
            self.canvas.itemconfig(self.right_eye, fill="#333333")
            # Thinking mouth (small o)
            self.canvas.coords(
                self.mouth,
                self.x - self.size//8, self.y + self.size//6,
                self.x + self.size//8, self.y + self.size//4
            )
            self.canvas.itemconfig(self.mouth, start=0, extent=360)
        elif mood == "excited":
            # Excited eyes (wide open black)
            self.canvas.itemconfig(self.left_eye, fill="#000000")
            self.canvas.itemconfig(self.right_eye, fill="#000000")
            # Excited mouth (huge smile)
            self.canvas.coords(
                self.mouth,
                self.x - self.size//2.5, self.y + self.size//8,
                self.x + self.size//2.5, self.y + self.size//1.8
            )
            self.canvas.itemconfig(self.mouth, start=0, extent=180)
        else:  # neutral
            self.canvas.itemconfig(self.left_eye, fill="#000000")
            self.canvas.itemconfig(self.right_eye, fill="#000000")
            self.canvas.coords(
                self.mouth,
                self.x - self.size//4, self.y + self.size//8,
                self.x + self.size//4, self.y + self.size//3
            )
            self.canvas.itemconfig(self.mouth, start=0, extent=180)
    
    def blink(self):
        # Temporary eye close
        self.canvas.itemconfig(self.left_eye, fill="#FFD93D")  # Same as head color
        self.canvas.itemconfig(self.right_eye, fill="#FFD93D")
        self.canvas.after(150, self.unblink)
    
    def unblink(self):
        if self.mood == "thinking":
            self.canvas.itemconfig(self.left_eye, fill="#333333")
            self.canvas.itemconfig(self.right_eye, fill="#333333")
        else:
            self.canvas.itemconfig(self.left_eye, fill="#000000")
            self.canvas.itemconfig(self.right_eye, fill="#000000")
    
    def animate(self):
        # Random blinking
        self.blink_counter += 1
        if self.blink_counter > random.randint(30, 80):
            self.blink()
            self.blink_counter = 0
        
        # Continue animation
        self.canvas.after(100, self.animate)

# --- Fancy Tkinter GUI ---
class HrudhiApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        self.robot_messages = [
            "Hi! I'm Hrudhi 😊 You can save anything in me and I'll help you retrieve your notes whenever you need them!",
            "Hello there! 🤖 I'm your personal AI memory assistant. Store your thoughts, and I'll find them instantly!",
            "Welcome! I'm Hrudhi - your friendly note-taking companion! 📝 Save your ideas and I'll remember them forever!",
            "Hi! I'm here to be your digital brain! 🧠 Tell me anything and I'll help you find it later with smart search!",
            "Greetings! I'm Hrudhi, your AI note buddy! ✨ I learn from everything you save and make finding notes super easy!"
        ]
        self.show_welcome_message()
        
    def setup_window(self):
        self.root.title("🤖 Hrudhi - Personal AI Agent")
        self.root.geometry("800x700")
        self.root.configure(bg="#1E1E2E")
        self.root.resizable(True, True)
        
        # Set minimum size
        self.root.minsize(600, 500)
        
        # Try to set window icon (if available)
        try:
            self.root.iconbitmap("robot.ico")
        except:
            pass
    
    def create_styles(self):
        # Configure modern, lightweight styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Modern color palette - cleaner and lighter
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
        style.configure('Modern.TLabel', 
                       background=self.colors['surface'], 
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 11))
        
        style.configure('Title.TLabel', 
                       background=self.colors['surface'], 
                       foreground=self.colors['primary'],
                       font=('Segoe UI', 20, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['surface'],
                       foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 10))
        
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
        
        style.map('Secondary.TButton',
                 background=[('active', '#F59E0B'),
                           ('pressed', '#D97706')])
    
    def create_widgets(self):
        # Modern main container with light theme
        self.root.configure(bg=self.colors['surface'])
        main_frame = tk.Frame(self.root, bg=self.colors['surface'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)
        
        # Clean header section
        header_frame = tk.Frame(main_frame, bg=self.colors['surface'])
        header_frame.pack(fill=tk.X, pady=(0, 24))
        
        # Smaller, refined robot
        self.robot_canvas = tk.Canvas(header_frame, width=60, height=60, 
                                    bg=self.colors['surface'], highlightthickness=0)
        self.robot_canvas.pack(side=tk.LEFT, padx=(0, 16))
        
        # Initialize smaller robot face
        self.robot = RobotFace(self.robot_canvas, 30, 30, 50)
        
        # Modern title section
        title_frame = tk.Frame(header_frame, bg=self.colors['surface'])
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(title_frame, text="Hrudhi", style='Title.TLabel')
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(title_frame, 
                                 text="Smart note companion", 
                                 style='Subtitle.TLabel')
        subtitle_label.pack(anchor=tk.W)
        
        # Lightweight status
        self.status_var = tk.StringVar(value="Ready to help")
        status_label = ttk.Label(title_frame, textvariable=self.status_var, 
                               style='Subtitle.TLabel')
        status_label.pack(anchor=tk.W, pady=(4, 0))
        
        # Modern card-based layout
        # Create notebook-style tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add Note tab
        add_frame = tk.Frame(notebook, bg=self.colors['surface'], padx=20, pady=20)
        notebook.add(add_frame, text="✏️ Add Note")
        
        # Clean input section
        self.text = tk.Text(add_frame, height=10, wrap=tk.WORD,
                           bg='white', fg=self.colors['text_primary'], 
                           insertbackground=self.colors['primary'],
                           font=('Segoe UI', 11), padx=16, pady=16,
                           relief=tk.FLAT, bd=1,
                           selectbackground=self.colors['primary'],
                           selectforeground='white')
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # Modern topic input
        topic_frame = tk.Frame(add_frame, bg=self.colors['surface'])
        topic_frame.pack(fill=tk.X, pady=(0, 16))
        
        tk.Label(topic_frame, text="Category:", bg=self.colors['surface'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        self.topic_entry = tk.Entry(topic_frame, bg='white', fg=self.colors['text_primary'],
                                  insertbackground=self.colors['primary'], font=('Segoe UI', 10),
                                  relief=tk.FLAT, bd=1)
        self.topic_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        self.topic_entry.insert(0, "ideas, meetings, reminders...")
        self.topic_entry.bind('<FocusIn>', self.clear_placeholder)
        
        # Modern save button
        save_btn = ttk.Button(add_frame, text="💾 Save Note", command=self.save_note, 
                            style='Modern.TButton')
        save_btn.pack(anchor=tk.E)
        
        # Search tab
        search_frame = tk.Frame(notebook, bg=self.colors['surface'], padx=20, pady=20)
        notebook.add(search_frame, text="🔍 Search")
        
        # Clean search input
        search_input_frame = tk.Frame(search_frame, bg=self.colors['surface'])
        search_input_frame.pack(fill=tk.X, pady=(0, 16))
        
        self.search_entry = tk.Entry(search_input_frame, bg='white', 
                                   fg=self.colors['text_primary'],
                                   insertbackground=self.colors['primary'], 
                                   font=('Segoe UI', 11), relief=tk.FLAT, bd=1)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.search_entry.bind('<Return>', lambda e: self.search_notes())
        
        search_btn = ttk.Button(search_input_frame, text="Search", 
                              command=self.search_notes, style='Secondary.TButton')
        search_btn.pack(side=tk.RIGHT)
        
        # Modern results display with scrollable frame
        results_frame = tk.Frame(search_frame, bg=self.colors['surface'])
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results canvas for custom styling
        self.results_canvas = tk.Canvas(results_frame, bg='white', highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                        command=self.results_canvas.yview)
        self.scrollable_frame = tk.Frame(self.results_canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.results_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.search_entry.bind('<Return>', lambda e: self.search_notes())
        
        search_btn = tk.Button(search_input_frame, text="🔎 Find it!", 
                              command=self.search_notes, bg="#FFD93D", fg="#000000",
                              font=('Arial', 11, 'bold'), relief='raised', bd=2,
                              activebackground="#FFC107", activeforeground="#000000",
                              cursor='hand2', padx=15, pady=5)
        search_btn.pack(side=tk.RIGHT, padx=(5, 5))
        
        # Results section - rounded display
        results_frame = tk.LabelFrame(main_frame, text="📋 Here's what I found for you", 
                                    bg="#2A2A3A", fg="#FFD93D",
                                    font=('Arial', 12, 'bold'), padx=15, pady=10,
                                    relief='groove', bd=3)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results listbox with scrollbar - rounded appearance
        results_list_frame = tk.Frame(results_frame, bg="#2A2A3A", relief='groove', bd=2)
        results_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results = tk.Listbox(results_list_frame, bg="#3A3A4A", fg="#FFFFFF",
                                selectbackground="#FFD93D", selectforeground="#000000", 
                                font=('Consolas', 10), relief=tk.FLAT, bd=0, 
                                activestyle='none', cursor='hand2')
        results_scrollbar = ttk.Scrollbar(results_list_frame, orient=tk.VERTICAL, 
                                        command=self.results.yview)
        self.results.configure(yscrollcommand=results_scrollbar.set)
        
        self.results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results.bind('<Double-1>', self.open_note)
        
    def clear_placeholder(self, event):
        if self.topic_entry.get() == "meeting notes, ideas, reminders...":
            self.topic_entry.delete(0, tk.END)
            self.topic_entry.configure(fg="#FFFFFF")
    
    def restore_placeholder(self, event):
        if not self.topic_entry.get():
            self.topic_entry.insert(0, "meeting notes, ideas, reminders...")
            self.topic_entry.configure(fg="#A6A6A6")
    
    def show_welcome_message(self):
        message = random.choice(self.robot_messages)
        self.status_var.set(message)
        self.robot.set_mood("happy")
    
    def update_status(self, message, mood="neutral"):
        self.status_var.set(message)
        self.robot.set_mood(mood)
    
    def save_note(self):
        text = self.text.get("1.0", tk.END).strip()
        topic = self.topic_entry.get().strip()
        
        if topic == "meeting notes, ideas, reminders...":
            topic = ""
            
        if not text or not topic:
            self.update_status("❌ Please enter both note and topic!", "thinking")
            messagebox.showwarning("Missing Data", "Please enter both note and topic.")
            return
        
        self.update_status("🔄 Saving note and learning...", "thinking")
        self.robot.set_mood("thinking")
        
        # Save in background thread
        def save_async():
            try:
                filename = save_note(text, topic)
                self.root.after(0, lambda: self.save_complete(filename))
            except Exception as e:
                self.root.after(0, lambda: self.save_error(str(e)))
        
        threading.Thread(target=save_async, daemon=True).start()
    
    def save_complete(self, filename):
        self.update_status(f"✅ Note saved! Hrudhi learned something new! 🎉", "excited")
        self.text.delete("1.0", tk.END)
        self.topic_entry.delete(0, tk.END)
        self.restore_placeholder(None)
        
        # Show success animation
        self.root.after(3000, lambda: self.update_status("Ready for your next note! 📝", "happy"))
    
    def save_error(self, error):
        self.update_status(f"❌ Error saving note: {error}", "neutral")
    
    def search_notes(self):
        query = self.search_entry.get().strip()
        if not query:
            self.update_status("❓ Please enter a search query!", "thinking")
            return
        
        self.update_status("🔍 Searching through my memory...", "thinking")
        self.results.delete(0, tk.END)
        self.results.insert(tk.END, "🔄 Searching...")
        
        # Search in background thread
        def search_async():
            try:
                results = search_notes(query)
                self.root.after(0, lambda: self.search_complete(results, query))
            except Exception as e:
                self.root.after(0, lambda: self.search_error(str(e)))
        
        threading.Thread(target=search_async, daemon=True).start()
    
    def search_complete(self, results, query):
        self.results.delete(0, tk.END)
        
        if not results:
            self.results.insert(tk.END, "🤔 No matching notes found")
            self.update_status("🤷‍♂️ No matches found. Try different keywords!", "thinking")
        else:
            self.update_status(f"🎯 Found {len(results)} matching notes!", "excited")
            for filename, similarity in results:
                score_bar = "█" * int(similarity * 10) + "░" * (10 - int(similarity * 10))
                display_name = filename.replace('.txt', '').replace('_', ' ')
                self.results.insert(tk.END, f"📄 {display_name}")
                self.results.insert(tk.END, f"   Match: {score_bar} {similarity:.1%}")
                self.results.insert(tk.END, "")  # Empty line for spacing
    
    def search_error(self, error):
        self.results.delete(0, tk.END)
        self.results.insert(tk.END, f"❌ Error: {error}")
        self.update_status("❌ Search error occurred", "neutral")
    
    def open_note(self, event):
        selection = self.results.curselection()
        if selection:
            selected_text = self.results.get(selection[0])
            if selected_text.startswith("📄"):
                # Extract filename from display text
                display_name = selected_text.replace("📄 ", "")
                
                # Find matching file
                for filename in os.listdir(NOTES_DIR):
                    if filename.endswith('.txt'):
                        file_display = filename.replace('.txt', '').replace('_', ' ')
                        if display_name in file_display:
                            filepath = os.path.join(NOTES_DIR, filename)
                            try:
                                with open(filepath, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                # Create fancy popup window
                                self.show_note_popup(filename, content)
                                self.update_status("📖 Showing your note!", "happy")
                                break
                            except Exception as e:
                                messagebox.showerror("Error", f"Could not open note: {e}")
    
    def show_note_popup(self, filename, content):
        popup = tk.Toplevel(self.root)
        popup.title(f"📄 {filename}")
        popup.geometry("600x400")
        popup.configure(bg="#1E1E2E")
        
        # Title with yellow theme
        title = tk.Label(popup, text=f"📄 {filename.replace('.txt', '').replace('_', ' ')}", 
                        bg="#1E1E2E", fg="#FFD93D", font=('Arial', 16, 'bold'))
        title.pack(pady=15)
        
        # Content with rounded appearance
        text_frame = tk.Frame(popup, bg="#1E1E2E", relief='groove', bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, bg="#3A3A4A", fg="#FFFFFF",
                            font=('Arial', 11), padx=15, pady=15, relief=tk.FLAT,
                            selectbackground="#FFD93D", selectforeground="#000000")
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert("1.0", content)
        text_widget.configure(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Close button with robot style
        close_btn = tk.Button(popup, text="✖️ Close", 
                            command=popup.destroy, bg="#FFD93D", fg="#000000",
                            font=('Arial', 12, 'bold'), relief='raised', bd=3,
                            activebackground="#FFC107", activeforeground="#000000",
                            cursor='hand2', padx=20, pady=8)
        close_btn.pack(pady=(0, 20))

if __name__ == "__main__":
    root = tk.Tk()
    app = HrudhiApp(root)
    root.mainloop()
