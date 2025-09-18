"""
Hrudhi: Personal AI Note-Taking Agent
Main entry point for the desktop client and core logic.
"""
import os
import datetime
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

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

def search_notes(query, top_k=5):
    query_embedding = model.encode(query).reshape(1, -1)
    results = []
    for filename, embedding in embeddings_db.items():
        similarity = cosine_similarity(query_embedding, [embedding])[0][0]
        results.append((filename, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# --- Tkinter GUI ---
class HrudhiApp:
    def __init__(self, root):
        self.root = root
        root.title("Hrudhi - Personal AI Agent")
        root.geometry("500x400")

        self.text = tk.Text(root, height=10)
        self.text.pack(pady=10)

        self.topic_entry = tk.Entry(root)
        self.topic_entry.pack(pady=5)
        self.topic_entry.insert(0, "Enter topic/context")

        self.save_btn = tk.Button(root, text="Save Note", command=self.save_note)
        self.save_btn.pack(pady=5)

        self.search_entry = tk.Entry(root)
        self.search_entry.pack(pady=5)
        self.search_entry.insert(0, "Search notes...")

        self.search_btn = tk.Button(root, text="Search", command=self.search_notes)
        self.search_btn.pack(pady=5)

        self.results = tk.Listbox(root)
        self.results.pack(fill=tk.BOTH, expand=True, pady=10)
        self.results.bind('<Double-1>', self.open_note)

    def save_note(self):
        text = self.text.get("1.0", tk.END).strip()
        topic = self.topic_entry.get().strip()
        if not text or not topic:
            messagebox.showwarning("Missing Data", "Please enter both note and topic.")
            return
        filename = save_note(text, topic)
        messagebox.showinfo("Note Saved", f"Saved as {filename}")
        self.text.delete("1.0", tk.END)
        self.topic_entry.delete(0, tk.END)

    def search_notes(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Missing Query", "Please enter a search query.")
            return
        results = search_notes(query)
        self.results.delete(0, tk.END)
        for filename, similarity in results:
            self.results.insert(tk.END, f"{filename} ({similarity:.2f})")

    def open_note(self, event):
        selection = self.results.curselection()
        if selection:
            filename = self.results.get(selection[0]).split(" (")[0]
            filepath = os.path.join(NOTES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            messagebox.showinfo(filename, content)

if __name__ == "__main__":
    root = tk.Tk()
    app = HrudhiApp(root)
    root.mainloop()
