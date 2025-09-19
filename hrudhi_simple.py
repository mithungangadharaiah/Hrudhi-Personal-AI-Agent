"""
Hrudhi Personal Assistant - Simple Working Version
Clean, minimal design with no problematic colors
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
from datetime import datetime
from pathlib import Path

class HrudhiSimple:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Hrudhi Personal Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.notes_file = Path("notes_simple.json")
        self.notes = self.load_notes()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create clean, simple UI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="🤖 Hrudhi Personal Assistant",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(0, 10))
        
        # Button frame
        btn_frame = tk.Frame(main_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(btn_frame, text="➕ New Note", command=self.new_note, 
                 bg='#4CAF50', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="💾 Save All", command=self.save_notes, 
                 bg='#2196F3', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🗑️ Delete", command=self.delete_note, 
                 bg='#f44336', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notes list (left side)
        list_frame = tk.Frame(content_frame, bg='#f0f0f0')
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        tk.Label(list_frame, text="📝 Your Notes", font=("Arial", 12, "bold"), 
                bg='#f0f0f0', fg='#333333').pack(anchor=tk.W, pady=(0, 5))
        
        # Notes listbox
        self.notes_listbox = tk.Listbox(list_frame, width=30, height=25, 
                                       bg='white', fg='#333333', 
                                       selectbackground='#e1f5fe')
        self.notes_listbox.pack(fill=tk.BOTH, expand=True)
        self.notes_listbox.bind('<<ListboxSelect>>', self.on_note_select)
        
        # Note editor (right side)
        editor_frame = tk.Frame(content_frame, bg='#f0f0f0')
        editor_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(editor_frame, text="✏️ Edit Note", font=("Arial", 12, "bold"), 
                bg='#f0f0f0', fg='#333333').pack(anchor=tk.W, pady=(0, 5))
        
        # Title entry
        tk.Label(editor_frame, text="Title:", bg='#f0f0f0', fg='#333333').pack(anchor=tk.W)
        self.title_entry = tk.Entry(editor_frame, font=("Arial", 12), bg='white')
        self.title_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Content text area
        tk.Label(editor_frame, text="Content:", bg='#f0f0f0', fg='#333333').pack(anchor=tk.W)
        self.content_text = scrolledtext.ScrolledText(
            editor_frame, 
            font=("Arial", 11), 
            bg='white', 
            fg='#333333',
            wrap=tk.WORD,
            height=20
        )
        self.content_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bg='#e0e0e0', fg='#333333', relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.refresh_notes_list()
    
    def load_notes(self):
        """Load notes from file"""
        if self.notes_file.exists():
            try:
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_notes(self):
        """Save all notes to file"""
        try:
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(self.notes, f, indent=2, ensure_ascii=False)
            self.status_var.set("✅ All notes saved successfully!")
            self.root.after(3000, lambda: self.status_var.set("Ready"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save notes: {str(e)}")
    
    def new_note(self):
        """Create a new note"""
        title = f"New Note {len(self.notes) + 1}"
        note = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "title": title,
            "content": "",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.notes.append(note)
        self.refresh_notes_list()
        self.notes_listbox.select_set(len(self.notes) - 1)
        self.load_selected_note()
        self.title_entry.focus()
        self.status_var.set("📝 New note created")
    
    def delete_note(self):
        """Delete selected note"""
        selection = self.notes_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a note to delete")
            return
        
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this note?"):
            index = selection[0]
            deleted_note = self.notes.pop(index)
            self.refresh_notes_list()
            self.clear_editor()
            self.status_var.set(f"🗑️ Deleted: {deleted_note['title']}")
    
    def refresh_notes_list(self):
        """Refresh the notes listbox"""
        self.notes_listbox.delete(0, tk.END)
        for note in self.notes:
            display_text = f"{note['title'][:25]}{'...' if len(note['title']) > 25 else ''}"
            self.notes_listbox.insert(tk.END, display_text)
    
    def on_note_select(self, event):
        """Handle note selection"""
        self.save_current_note()
        self.load_selected_note()
    
    def load_selected_note(self):
        """Load selected note into editor"""
        selection = self.notes_listbox.curselection()
        if selection:
            index = selection[0]
            note = self.notes[index]
            
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, note['title'])
            
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(1.0, note['content'])
            
            self.status_var.set(f"📖 Loaded: {note['title']}")
    
    def save_current_note(self):
        """Save current note being edited"""
        selection = self.notes_listbox.curselection()
        if selection:
            index = selection[0]
            self.notes[index]['title'] = self.title_entry.get()
            self.notes[index]['content'] = self.content_text.get(1.0, tk.END).strip()
            self.notes[index]['modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.refresh_notes_list()
            self.notes_listbox.select_set(index)
    
    def clear_editor(self):
        """Clear the editor"""
        self.title_entry.delete(0, tk.END)
        self.content_text.delete(1.0, tk.END)
    
    def run(self):
        """Start the application"""
        # Auto-save every 30 seconds
        def auto_save():
            self.save_current_note()
            self.save_notes()
            self.root.after(30000, auto_save)
        
        self.root.after(30000, auto_save)
        
        # Handle window close
        def on_closing():
            self.save_current_note()
            self.save_notes()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = HrudhiSimple()
    app.run()