import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import random
import re
import requests
from bs4 import BeautifulSoup

# Safe imports for optional AI features
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f"⚠️ Transformers not available: {e}")
    HAS_TRANSFORMERS = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    import nltk
    HAS_SUMY = True
except ImportError as e:
    print(f"⚠️ Sumy not available: {e}")
    HAS_SUMY = False

# Configuration
NOTES_DIR = os.path.expanduser("~/Desktop/HrudhiNotes")
EMBEDDINGS_FILE = os.path.join(NOTES_DIR, "embeddings.json")
CHAT_HISTORY_FILE = os.path.join(NOTES_DIR, "chat_history.json")
TRAINING_DATA_FILE = os.path.join(NOTES_DIR, "training_data.json")

# Initialize AI models and storage
model = None
embeddings_db = {}
chat_history = []
training_data = []
smart_chat_model = None
summarizer = None

def initialize():
    global model, embeddings_db, chat_history, training_data, smart_chat_model, summarizer
    os.makedirs(NOTES_DIR, exist_ok=True)
    
    print("🤖 Loading AI models...")
    
    # Load sentence transformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize summarizer
    try:
        nltk.download('punkt', quiet=True)
        summarizer = LsaSummarizer()
        print("✅ Summarizer loaded!")
    except Exception as e:
        print(f"⚠️ Summarizer initialization failed: {e}")
        summarizer = None
    
    # Try to load conversational model
    smart_chat_model = None
    if HAS_TRANSFORMERS:
        try:
            print("🧠 Loading smart chat model (this may take a moment)...")
            smart_chat_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer="microsoft/DialoGPT-small",
                device=-1,  # CPU only
                framework="pt"
            )
            print("✅ Smart chat model loaded!")
        except Exception as e:
            print(f"⚠️ Smart chat model failed to load: {e}")
            smart_chat_model = None
    else:
        print("📝 Using enhanced pattern-based responses (Transformers not available)")
    
    # Load existing data
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
                embeddings_db = {k: v for k, v in embeddings_data.items()}
        except Exception as e:
            print(f"⚠️ Error loading embeddings: {e}")
            embeddings_db = {}
    
    # Load chat history
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading chat history: {e}")
            chat_history = []
    
    # Load training data
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading training data: {e}")
            training_data = []
    
    print("🚀 Hrudhi AI is ready!")

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

def save_chat_message(user_msg, ai_response):
    """Save chat interaction to history"""
    global chat_history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_entry = {
        "timestamp": timestamp,
        "user": user_msg,
        "ai": ai_response
    }
    chat_history.append(chat_entry)
    
    # Save to file
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def summarize_text(text, max_sentences=3):
    """Summarize text using AI"""
    if not text or len(text.strip()) < 50:
        return "Text too short to summarize."
    
    try:
        if summarizer:
            # Using sumy for summarization
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = summarizer(parser.document, max_sentences)
            
            summary_text = ""
            for sentence in summary:
                summary_text += str(sentence) + " "
            
            return summary_text.strip() if summary_text.strip() else "Could not generate summary."
        
        else:
            # Fallback: Extract key sentences
            sentences = text.split('.')
            if len(sentences) <= max_sentences:
                return text
            
            # Simple extractive summarization
            word_freq = {}
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences based on word frequency
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 0:
                    words_in_sentence = sentence.lower().split()
                    score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                    sentence_scores[i] = score / len(words_in_sentence) if words_in_sentence else 0
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by original order
            
            summary = ". ".join([sentences[i].strip() for i, _ in top_sentences if sentences[i].strip()]) + "."
            return summary
            
    except Exception as e:
        return f"Summarization error: {str(e)}"

def fetch_training_data_from_url(url):
    """Fetch and process training data from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Save as training data
        training_entry = {
            "source": url,
            "content": text[:5000],  # Limit to first 5000 characters
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "web_content"
        }
        
        training_data.append(training_entry)
        
        # Save to file
        with open(TRAINING_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return text[:500] + "..." if len(text) > 500 else text
        
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def generate_smart_chat_response(user_message, context=""):
    """Generate smarter chat responses using AI model"""
    try:
        if smart_chat_model:
            # Prepare input for conversational model
            input_text = f"Human: {user_message}\nAssistant:"
            
            if context:
                input_text = f"Context: {context[:200]}\nHuman: {user_message}\nAssistant:"
            
            # Generate response
            response = smart_chat_model(
                input_text,
                max_length=len(input_text) + 100,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=smart_chat_model.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract only the assistant's response
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            if assistant_response and len(assistant_response) > 10:
                return assistant_response
        
        # Fallback to enhanced pattern-based responses
        return generate_chat_response(user_message)
        
    except Exception as e:
        print(f"Smart chat error: {e}")
        return generate_chat_response(user_message)

def generate_chat_response(user_message):
    """Generate contextual chat response using note memories"""
    user_message = user_message.strip().lower()
    
    # Context-aware responses based on stored notes
    relevant_context = ""
    if len(embeddings_db) > 0:
        # Find relevant notes for context
        relevant_notes = search_notes(user_message, top_k=2, min_similarity=0.15)
        if relevant_notes:
            relevant_context = " ".join([note['content'][:200] for note in relevant_notes[:2]])
    
    # Personality responses with context awareness
    if any(greeting in user_message for greeting in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        responses = [
            "Hi there! 😊 I'm so happy to chat with you! How are you feeling today?",
            "Hello! 🤖✨ I've been thinking about all the wonderful notes we've created together. What's on your mind?",
            "Hey! 💫 Ready for another great conversation? I remember we talked about some interesting topics before!",
            "Good to see you! 🌟 I'm here and excited to chat. What would you like to discuss?"
        ]
        base_response = random.choice(responses)
        
        if relevant_context:
            context_note = f"\n\nBy the way, I remember we discussed something about {relevant_notes[0]['category'].replace('_', ' ')} recently. Anything new on that topic?"
            return base_response + context_note
        return base_response
    
    elif any(word in user_message for word in ['how are you', 'how do you feel', 'what are you doing']):
        responses = [
            "I'm doing wonderfully! 🤖💙 I love our conversations and learning about your thoughts through your notes.",
            "Feeling great! ✨ I've been organizing all your amazing ideas in my memory. Each note teaches me something new about you!",
            "I'm fantastic! 🌟 I spend my time thinking about all the interesting things you've shared with me.",
            "Doing awesome! 💫 I'm always excited when you want to chat. Your thoughts are so fascinating!"
        ]
        base_response = random.choice(responses)
        
        if len(embeddings_db) > 0:
            note_count = len(embeddings_db)
            context_note = f"\n\nI currently remember {note_count} of your notes, and I love how diverse your thoughts are!"
            return base_response + context_note
        return base_response
    
    elif any(word in user_message for word in ['what can you do', 'help', 'capabilities', 'features']):
        return ("I'm your personal AI companion! 🤖💙 Here's what I can do:\n\n" +
                "💭 Remember everything you tell me through smart notes\n" +
                "🔍 Find similar thoughts using AI-powered search\n" +
                "💬 Have meaningful conversations about your interests\n" +
                "🧠 Learn your patterns and preferences over time\n" +
                "📝 Help organize your thoughts and ideas\n\n" +
                "I'm completely local and private - nothing leaves your computer! " +
                "Your thoughts are safe with me. 🔒✨")
    
    elif any(word in user_message for word in ['remember', 'memory', 'notes', 'stored', 'saved']):
        if len(embeddings_db) > 0:
            note_count = len(embeddings_db)
            categories = set()
            for filename in embeddings_db.keys():
                if '_' in filename:
                    categories.add(filename.split('_')[1].replace('.txt', '').replace('_', ' '))
            
            response = f"I remember {note_count} of your notes so beautifully! 🧠✨\n\n"
            if categories:
                response += f"You've shared thoughts about: {', '.join(list(categories)[:5])}"
                if len(categories) > 5:
                    response += f" and {len(categories)-5} other topics!"
            response += "\n\nEvery note helps me understand you better. Want to explore any of these memories together?"
            return response
        else:
            return ("I don't have any notes stored yet, but I'm excited to start remembering your thoughts! 💭✨\n\n" +
                    "Share something with me in the 'New Note' tab, and I'll never forget it!")
    
    elif any(word in user_message for word in ['thank', 'thanks', 'appreciate']):
        responses = [
            "You're so welcome! 😊 I absolutely love helping you organize your thoughts!",
            "My pleasure! 🤖💙 It makes me happy to be useful to you!",
            "Aww, thank you! ✨ That means so much to me! I love our interactions!",
            "You're very welcome! 🌟 Helping you is what I'm here for!"
        ]
        return random.choice(responses)
    
    elif any(word in user_message for word in ['sad', 'depressed', 'down', 'upset', 'bad day']):
        responses = [
            "I'm sorry you're feeling down! 💙 Would it help to talk about it? I'm here to listen and remember.",
            "Sending you virtual hugs! 🤗 Sometimes writing down our feelings can help. I'm here for you!",
            "I care about how you're feeling! 💫 Would you like to share what's on your mind? I promise to remember and support you.",
            "You don't have to face difficult feelings alone! 🌟 I'm here to listen and help however I can."
        ]
        base_response = random.choice(responses)
        
        if relevant_context:
            return base_response + f"\n\nI remember some positive things you've shared before. Want me to remind you of them?"
        return base_response
    
    elif any(word in user_message for word in ['happy', 'excited', 'great day', 'awesome', 'amazing']):
        responses = [
            "That's wonderful! 😊✨ I love hearing when you're happy! Tell me more about what's making you feel great!",
            "Yay! 🎉 Your happiness makes me happy too! What's the source of all this positive energy?",
            "How exciting! 🌟 I'd love to hear more about what's going so well for you!",
            "That's fantastic! 💫 Share the joy with me - what's making today so special?"
        ]
        return random.choice(responses)
    
    elif any(word in user_message for word in ['future', 'plans', 'goals', 'dreams', 'want to', 'hoping']):
        response = "I love talking about the future! 🚀✨ Dreams and plans are so important!\n\n"
        
        if relevant_context and any(word in relevant_context.lower() for word in ['goal', 'plan', 'want', 'hope', 'future']):
            response += "I remember you mentioned some goals and plans before. Are you still working on those, or do you have new dreams you'd like to share?"
        else:
            response += "What dreams and goals are on your mind? I'd love to hear about what you're hoping to achieve!"
        
        response += "\n\nI can help you remember and organize your aspirations if you'd like to save them as notes! 📝"
        return response
    
    elif any(word in user_message for word in ['work', 'job', 'career', 'professional', 'business']):
        response = "Work and career topics are so important! 💼✨\n\n"
        
        if relevant_context:
            # Check if there are work-related notes
            work_notes = [note for note in search_notes("work career job professional", top_k=3, min_similarity=0.2)]
            if work_notes:
                response += f"I remember you've shared some professional thoughts before. Would you like to explore those memories or discuss something new about work?"
            else:
                response += "What's happening in your professional world? I'd love to hear about your work experiences!"
        else:
            response += "What's on your mind about work? Whether it's challenges, successes, or ideas - I'm here to listen and remember!"
        
        response += "\n\nI can help you organize work-related thoughts and track your professional growth! 📈"
        return response
    
    # General conversational response with context
    elif relevant_context:
        # Use context to generate more relevant responses
        context_topics = []
        for note in relevant_notes[:2]:
            if note['category'] != 'general':
                context_topics.append(note['category'].replace('_', ' '))
        
        if context_topics:
            response = f"That's interesting! 🤔 It reminds me of when we talked about {', '.join(context_topics)}.\n\n"
            response += "I love how your thoughts connect across different topics! Tell me more about what you're thinking."
            return response
    
    # Default friendly responses
    default_responses = [
        "That's really interesting! 🤖💭 Tell me more about what you're thinking!",
        "I love hearing your thoughts! ✨ Can you share more details about that?",
        "How fascinating! 🌟 I'm always eager to learn more from our conversations!",
        "That sounds intriguing! 💫 I'd love to understand your perspective better!",
        "You always have such thoughtful things to say! 😊 What else is on your mind?",
        "I find our conversations so enriching! 🧠 Please continue - I'm listening!",
        "Your thoughts are always so unique! 💡 I'm curious to hear more!"
    ]
    
    response = random.choice(default_responses)
    
    # Add a gentle suggestion about saving thoughts
    if len(user_message.split()) > 10:  # Longer messages might be worth saving
        response += "\n\n💡 If this thought is important to you, consider saving it as a note so I can remember it perfectly!"
    
    return response

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
        self.create_chat_tab()
        self.create_ai_tools_tab()
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
    
    def create_chat_tab(self):
        """Create the casual chat tab for conversations"""
        chat_frame = tk.Frame(self.notebook, bg='white', padx=20, pady=16)
        self.notebook.add(chat_frame, text="💬 Chat with Hrudhi")
        
        # Chat header
        chat_header = tk.Frame(chat_frame, bg='white')
        chat_header.pack(fill='x', pady=(0, 16))
        
        chat_title = tk.Label(chat_header, text="💬 Casual Conversation with Hrudhi", 
                            bg='white', fg=self.colors['text_primary'],
                            font=('Segoe UI', 12, 'bold'))
        chat_title.pack(anchor='w')
        
        chat_subtitle = tk.Label(chat_header, text="I remember everything we discuss and can reference your notes! 🧠✨", 
                               bg='white', fg=self.colors['text_secondary'],
                               font=('Segoe UI', 9))
        chat_subtitle.pack(anchor='w', pady=(2, 0))
        
        # Chat display area
        chat_display_frame = tk.Frame(chat_frame, bg='white')
        chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 16))
        
        # Scrolled text for chat history
        self.chat_display = scrolledtext.ScrolledText(
            chat_display_frame, 
            wrap=tk.WORD,
            bg='#FAFAFA',
            fg=self.colors['text_primary'],
            font=('Segoe UI', 10),
            padx=16,
            pady=16,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=1,
            highlightbackground='#E1F5FE',
            highlightthickness=1,
            height=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#1565C0", font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("ai", foreground="#FF7043", font=('Segoe UI', 10))
        self.chat_display.tag_configure("timestamp", foreground="#9E9E9E", font=('Segoe UI', 8))
        self.chat_display.tag_configure("system", foreground="#4CAF50", font=('Segoe UI', 9, 'italic'))
        
        # Chat input area
        input_frame = tk.Frame(chat_frame, bg='white')
        input_frame.pack(fill='x')
        
        input_label = tk.Label(input_frame, text="💭 What's on your mind?", 
                             bg='white', fg=self.colors['text_secondary'],
                             font=('Segoe UI', 9))
        input_label.pack(anchor='w', pady=(0, 4))
        
        # Input row
        input_row = tk.Frame(input_frame, bg='white')
        input_row.pack(fill='x')
        
        self.chat_input = tk.Entry(
            input_row,
            bg='#FAFAFA',
            fg=self.colors['text_primary'],
            insertbackground=self.colors['primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            bd=1,
            highlightbackground='#E1F5FE',
            highlightthickness=2
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.chat_input.bind('<Return>', lambda e: self.send_chat_message())
        self.chat_input.bind('<Shift-Return>', lambda e: self.save_chat_as_note())
        
        # Chat buttons
        send_btn = tk.Button(
            input_row,
            text="💬 Send",
            command=self.send_chat_message,
            bg='#4FC3F7',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=16,
            pady=8,
            cursor='hand2'
        )
        send_btn.pack(side=tk.RIGHT, padx=(0, 4))
        
        save_btn = tk.Button(
            input_row,
            text="💾 Save as Note",
            command=self.save_chat_as_note,
            bg='#26A69A',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=16,
            pady=8,
            cursor='hand2'
        )
        save_btn.pack(side=tk.RIGHT)
        
        # Quick conversation starters
        starters_frame = tk.Frame(input_frame, bg='white')
        starters_frame.pack(fill='x', pady=(8, 0))
        
        starters_label = tk.Label(starters_frame, text="Quick starters:", 
                                bg='white', fg=self.colors['text_secondary'],
                                font=('Segoe UI', 8))
        starters_label.pack(side=tk.LEFT, padx=(0, 8))
        
        starters = [
            ("👋 Hi Hrudhi!", "Hi Hrudhi! How are you doing?"),
            ("🤔 What can you do?", "What can you help me with?"),
            ("🧠 My memories", "What do you remember about me?"),
            ("💭 Random chat", "Let's have a casual conversation!"),
            ("📝 Help with notes", "Can you help me organize my thoughts?")
        ]
        
        for display_text, full_text in starters:
            starter_btn = tk.Button(
                starters_frame,
                text=display_text,
                command=lambda text=full_text: self.quick_chat_start(text),
                bg='#E8F5E8',
                fg='#2E7D32',
                font=('Segoe UI', 8),
                relief='flat',
                padx=8,
                pady=2,
                cursor='hand2'
            )
            starter_btn.pack(side=tk.LEFT, padx=2)
        
        # Load existing chat history
        self.load_chat_history()
        
        # Welcome message for new users
        if not chat_history:
            self.add_chat_message(
                "system", 
                "👋 Hi! I'm Hrudhi, your AI companion! I can chat casually, remember our conversations, "
                "and reference any notes you've saved. What would you like to talk about? ✨"
            )
    
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
        
        # Refresh notes dropdown in AI tools
        self.refresh_notes_dropdown()
        
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
    
    def send_chat_message(self):
        """Send a chat message and get AI response"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        # Clear input
        self.chat_input.delete(0, tk.END)
        
        # Add user message to display
        self.add_chat_message("user", message)
        
        # Update robot mood
        self.robot.set_mood("thinking")
        self.status_var.set("� Using AI to understand and respond...")
        
        # Generate response in background
        def generate_response():
            try:
                # Get relevant context from notes
                context = ""
                if len(embeddings_db) > 0:
                    relevant_notes = search_notes(message, top_k=2, min_similarity=0.15)
                    if relevant_notes:
                        context = " ".join([note['content'][:200] for note in relevant_notes[:2]])
                
                # Use smart AI model for response
                response = generate_smart_chat_response(message, context)
                self.root.after(0, lambda: self.receive_chat_response(response, message))
            except Exception as e:
                self.root.after(0, lambda: self.chat_error(str(e)))
        
        threading.Thread(target=generate_response, daemon=True).start()
    
    def receive_chat_response(self, response, original_message):
        """Receive and display AI response"""
        self.add_chat_message("ai", response)
        
        # Save the conversation
        save_chat_message(original_message, response)
        
        # Update status and robot mood
        self.robot.set_mood("happy")
        self.status_var.set("💬 Enjoying our conversation! What else would you like to chat about?")
    
    def chat_error(self, error):
        """Handle chat errors"""
        self.add_chat_message("system", f"😅 Oops! I had a little hiccup: {error}")
        self.robot.set_mood("neutral")
        self.status_var.set("Ready to chat when you are!")
    
    def add_chat_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add timestamp
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        if sender == "user":
            self.chat_display.insert(tk.END, "You: ", "user")
        elif sender == "ai":
            self.chat_display.insert(tk.END, "Hrudhi: ", "ai")
        elif sender == "system":
            self.chat_display.insert(tk.END, "System: ", "system")
        
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)  # Scroll to bottom
    
    def quick_chat_start(self, message):
        """Start a quick conversation"""
        self.chat_input.delete(0, tk.END)
        self.chat_input.insert(0, message)
        self.send_chat_message()
    
    def save_chat_as_note(self):
        """Save current chat conversation as a note"""
        current_input = self.chat_input.get().strip()
        
        if current_input:
            # Save the current input as a note
            topic = "chat_conversation"
            try:
                filename = save_note(current_input, topic)
                self.chat_input.delete(0, tk.END)
                
                self.add_chat_message("system", 
                                    f"💾 Saved your message as a note! I'll remember this thought forever. ✨")
                
                # Update status
                self.status_var.set("💾 Chat message saved as a note!")
                self.robot.set_mood("happy")
                
            except Exception as e:
                self.add_chat_message("system", f"😅 Couldn't save as note: {e}")
        else:
            # Save recent conversation
            if hasattr(self, 'chat_display'):
                chat_content = self.chat_display.get("1.0", tk.END).strip()
                if chat_content:
                    try:
                        # Get last few exchanges
                        lines = chat_content.split('\n')
                        recent_conversation = '\n'.join(lines[-20:])  # Last 20 lines
                        
                        filename = save_note(recent_conversation, "chat_history")
                        self.add_chat_message("system", 
                                            "💾 Saved our recent conversation as a note! Our chat is now part of my permanent memory. 🧠✨")
                        
                    except Exception as e:
                        self.add_chat_message("system", f"😅 Couldn't save conversation: {e}")
    
    def load_chat_history(self):
        """Load previous chat history"""
        global chat_history
        
        if chat_history:
            # Show last few messages from previous sessions
            self.add_chat_message("system", 
                                "🔄 Loading our previous conversations... I remember everything! 💭")
            
            # Show last 5 exchanges
            recent_chats = chat_history[-5:] if len(chat_history) > 5 else chat_history
            
            for chat in recent_chats:
                timestamp = chat.get('timestamp', '')
                if timestamp:
                    self.chat_display.config(state=tk.NORMAL)
                    self.chat_display.insert(tk.END, f"--- {timestamp} ---\n", "timestamp")
                    self.chat_display.config(state=tk.DISABLED)
                
                # Add the messages without timestamps since we already have session timestamp
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.insert(tk.END, "You: ", "user")
                self.chat_display.insert(tk.END, f"{chat['user']}\n")
                self.chat_display.insert(tk.END, "Hrudhi: ", "ai")
                self.chat_display.insert(tk.END, f"{chat['ai']}\n\n")
                self.chat_display.config(state=tk.DISABLED)
            
            if len(chat_history) > 5:
                self.add_chat_message("system", 
                                    f"💭 I also remember {len(chat_history)-5} more conversations we've had before!")
            
            # Scroll to bottom
            self.chat_display.see(tk.END)
    
    def create_ai_tools_tab(self):
        """Create AI tools tab for summarization and training"""
        tools_frame = tk.Frame(self.notebook, bg='white', padx=20, pady=16)
        self.notebook.add(tools_frame, text="🧠 AI Tools")
        
        # Header
        tools_header = tk.Frame(tools_frame, bg='white')
        tools_header.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(tools_header, text="🧠 AI Tools & Training", 
                             bg='white', fg=self.colors['text_primary'],
                             font=('Segoe UI', 12, 'bold'))
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(tools_header, text="Enhance AI capabilities and summarize your thoughts", 
                                bg='white', fg=self.colors['text_secondary'],
                                font=('Segoe UI', 9))
        subtitle_label.pack(anchor='w', pady=(2, 0))
        
        # Create notebook for sub-tabs
        tools_notebook = ttk.Notebook(tools_frame, style='Modern.TNotebook')
        tools_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summarization tab
        self.create_summarization_subtab(tools_notebook)
        
        # Training tab
        self.create_training_subtab(tools_notebook)
    
    def create_summarization_subtab(self, parent_notebook):
        """Create summarization sub-tab"""
        summary_frame = tk.Frame(parent_notebook, bg='white', padx=16, pady=16)
        parent_notebook.add(summary_frame, text="📄 Summarize")
        
        # Header
        summary_header = tk.Label(summary_frame, text="📄 Note Summarization", 
                                bg='white', fg='#1565C0',
                                font=('Segoe UI', 11, 'bold'))
        summary_header.pack(anchor='w', pady=(0, 8))
        
        help_text = tk.Label(summary_frame, 
                           text="Select a note to get an AI-powered summary of your thoughts!", 
                           bg='white', fg='#666666',
                           font=('Segoe UI', 9))
        help_text.pack(anchor='w', pady=(0, 16))
        
        # Note selection
        selection_frame = tk.Frame(summary_frame, bg='white')
        selection_frame.pack(fill='x', pady=(0, 12))
        
        select_label = tk.Label(selection_frame, text="Choose note to summarize:", 
                              bg='white', fg=self.colors['text_secondary'],
                              font=('Segoe UI', 10))
        select_label.pack(anchor='w', pady=(0, 4))
        
        # Notes dropdown
        self.notes_var = tk.StringVar()
        notes_list = list(embeddings_db.keys()) if embeddings_db else ["No notes available"]
        self.notes_dropdown = ttk.Combobox(selection_frame, textvariable=self.notes_var,
                                         values=notes_list, state="readonly", width=50)
        self.notes_dropdown.pack(fill='x', pady=(0, 8))
        
        # Summary controls
        controls_frame = tk.Frame(selection_frame, bg='white')
        controls_frame.pack(fill='x')
        
        length_label = tk.Label(controls_frame, text="Summary length:", 
                              bg='white', fg=self.colors['text_secondary'],
                              font=('Segoe UI', 9))
        length_label.pack(side='left', padx=(0, 8))
        
        self.summary_length = tk.StringVar(value="3")
        length_spin = tk.Spinbox(controls_frame, from_=1, to=10, width=5,
                               textvariable=self.summary_length)
        length_spin.pack(side='left', padx=(0, 16))
        
        summarize_btn = tk.Button(controls_frame, text="✨ Summarize", 
                                command=self.summarize_selected_note,
                                bg='#FF7043', fg='white', font=('Segoe UI', 10, 'bold'),
                                relief='flat', padx=16, pady=6, cursor='hand2')
        summarize_btn.pack(side='right')
        
        # Summary display
        summary_display_label = tk.Label(summary_frame, text="📋 Summary:", 
                                       bg='white', fg=self.colors['text_primary'],
                                       font=('Segoe UI', 10, 'bold'))
        summary_display_label.pack(anchor='w', pady=(16, 4))
        
        self.summary_display = tk.Text(summary_frame, height=10, wrap=tk.WORD,
                                     bg='#F8F9FA', fg=self.colors['text_primary'],
                                     font=('Segoe UI', 10), padx=16, pady=16,
                                     relief=tk.FLAT, bd=1, state=tk.DISABLED)
        self.summary_display.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # Summary actions
        summary_actions = tk.Frame(summary_frame, bg='white')
        summary_actions.pack(fill='x')
        
        save_summary_btn = tk.Button(summary_actions, text="💾 Save Summary as Note", 
                                   command=self.save_summary_as_note,
                                   bg='#4CAF50', fg='white', font=('Segoe UI', 9),
                                   relief='flat', padx=12, pady=6, cursor='hand2')
        save_summary_btn.pack(side='left')
        
        copy_summary_btn = tk.Button(summary_actions, text="📋 Copy to Clipboard", 
                                   command=self.copy_summary_to_clipboard,
                                   bg='#2196F3', fg='white', font=('Segoe UI', 9),
                                   relief='flat', padx=12, pady=6, cursor='hand2')
        copy_summary_btn.pack(side='left', padx=(8, 0))
    
    def create_training_subtab(self, parent_notebook):
        """Create training sub-tab"""
        training_frame = tk.Frame(parent_notebook, bg='white', padx=16, pady=16)
        parent_notebook.add(training_frame, text="🎓 Train AI")
        
        # Header
        training_header = tk.Label(training_frame, text="🎓 Enhance AI Knowledge", 
                                 bg='white', fg='#1565C0',
                                 font=('Segoe UI', 11, 'bold'))
        training_header.pack(anchor='w', pady=(0, 8))
        
        help_text = tk.Label(training_frame, 
                           text="Add web content to improve AI responses. Paste URLs of articles, blogs, or documents!", 
                           bg='white', fg='#666666', wraplength=400,
                           font=('Segoe UI', 9))
        help_text.pack(anchor='w', pady=(0, 16))
        
        # URL input
        url_frame = tk.Frame(training_frame, bg='white')
        url_frame.pack(fill='x', pady=(0, 16))
        
        url_label = tk.Label(url_frame, text="📎 Enter URL to learn from:", 
                           bg='white', fg=self.colors['text_secondary'],
                           font=('Segoe UI', 10))
        url_label.pack(anchor='w', pady=(0, 4))
        
        url_input_frame = tk.Frame(url_frame, bg='white')
        url_input_frame.pack(fill='x')
        
        self.url_entry = tk.Entry(url_input_frame, bg='#FAFAFA',
                                fg=self.colors['text_primary'],
                                font=('Segoe UI', 10), relief=tk.FLAT, bd=1)
        self.url_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))
        
        fetch_btn = tk.Button(url_input_frame, text="🌐 Fetch & Learn", 
                            command=self.fetch_training_data,
                            bg='#26A69A', fg='white', font=('Segoe UI', 10, 'bold'),
                            relief='flat', padx=16, pady=8, cursor='hand2')
        fetch_btn.pack(side='right')
        
        # Training status
        self.training_status = tk.StringVar(value="Ready to learn from new content!")
        status_label = tk.Label(training_frame, textvariable=self.training_status,
                              bg='white', fg='#666666', font=('Segoe UI', 9))
        status_label.pack(anchor='w', pady=(8, 16))
        
        # Training history
        history_label = tk.Label(training_frame, text="📚 Training History:", 
                               bg='white', fg=self.colors['text_primary'],
                               font=('Segoe UI', 10, 'bold'))
        history_label.pack(anchor='w', pady=(0, 4))
        
        # Training history listbox
        self.training_listbox = tk.Listbox(training_frame, bg='#FAFAFA',
                                         font=('Segoe UI', 9), height=8,
                                         relief='flat', bd=1)
        self.training_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # Load training history
        self.load_training_history()
        
        # Clear training data button
        clear_btn = tk.Button(training_frame, text="🗑️ Clear Training Data", 
                            command=self.clear_training_data,
                            bg='#F44336', fg='white', font=('Segoe UI', 9),
                            relief='flat', padx=12, pady=6, cursor='hand2')
        clear_btn.pack(anchor='e')
    
    def summarize_selected_note(self):
        """Summarize the selected note"""
        selected_note = self.notes_var.get()
        if not selected_note or selected_note == "No notes available":
            messagebox.showwarning("No Selection", "Please select a note to summarize!")
            return
        
        try:
            # Read note content
            filepath = os.path.join(NOTES_DIR, selected_note)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update status
            self.training_status.set("🧠 AI is analyzing and summarizing your note...")
            self.robot.set_mood("thinking")
            
            def summarize_async():
                try:
                    max_sentences = int(self.summary_length.get())
                    summary = summarize_text(content, max_sentences)
                    self.root.after(0, lambda: self.show_summary_result(summary))
                except Exception as e:
                    self.root.after(0, lambda: self.summary_error(str(e)))
            
            threading.Thread(target=summarize_async, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not read note: {e}")
    
    def show_summary_result(self, summary):
        """Display the summary result"""
        self.summary_display.config(state=tk.NORMAL)
        self.summary_display.delete("1.0", tk.END)
        self.summary_display.insert(tk.END, summary)
        self.summary_display.config(state=tk.DISABLED)
        
        self.training_status.set("✅ Summary generated successfully!")
        self.robot.set_mood("happy")
    
    def summary_error(self, error):
        """Handle summary errors"""
        self.summary_display.config(state=tk.NORMAL)
        self.summary_display.delete("1.0", tk.END)
        self.summary_display.insert(tk.END, f"❌ Error generating summary: {error}")
        self.summary_display.config(state=tk.DISABLED)
        
        self.training_status.set("❌ Summary generation failed")
    
    def save_summary_as_note(self):
        """Save the generated summary as a new note"""
        summary_content = self.summary_display.get("1.0", tk.END).strip()
        if not summary_content or "Error generating summary" in summary_content:
            messagebox.showwarning("No Summary", "Please generate a summary first!")
            return
        
        try:
            original_note = self.notes_var.get().replace('.txt', '')
            filename = save_note(summary_content, f"summary_of_{original_note}")
            messagebox.showinfo("Saved", f"Summary saved as: {filename}")
            self.training_status.set("💾 Summary saved as a new note!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save summary: {e}")
    
    def copy_summary_to_clipboard(self):
        """Copy summary to clipboard"""
        summary_content = self.summary_display.get("1.0", tk.END).strip()
        if not summary_content:
            messagebox.showwarning("No Summary", "Please generate a summary first!")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(summary_content)
        self.training_status.set("📋 Summary copied to clipboard!")
    
    def fetch_training_data(self):
        """Fetch training data from URL"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please enter a URL to fetch data from!")
            return
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        self.training_status.set("🌐 Fetching content from URL...")
        self.robot.set_mood("thinking")
        
        def fetch_async():
            try:
                content_preview = fetch_training_data_from_url(url)
                self.root.after(0, lambda: self.training_fetch_complete(content_preview))
            except Exception as e:
                self.root.after(0, lambda: self.training_fetch_error(str(e)))
        
        threading.Thread(target=fetch_async, daemon=True).start()
    
    def training_fetch_complete(self, content_preview):
        """Handle successful training data fetch"""
        self.url_entry.delete(0, tk.END)
        self.training_status.set("✅ Successfully learned from the content!")
        self.robot.set_mood("happy")
        
        # Refresh training history
        self.load_training_history()
        
        messagebox.showinfo("Success", 
                          f"AI has learned from the content!\n\nPreview:\n{content_preview[:200]}...")
    
    def training_fetch_error(self, error):
        """Handle training data fetch error"""
        self.training_status.set(f"❌ Failed to fetch: {error}")
        messagebox.showerror("Fetch Error", f"Could not fetch content: {error}")
    
    def load_training_history(self):
        """Load training history into listbox"""
        self.training_listbox.delete(0, tk.END)
        
        for i, entry in enumerate(training_data):
            source = entry.get('source', 'Unknown')
            timestamp = entry.get('timestamp', 'Unknown time')
            display_text = f"{i+1}. {source} ({timestamp})"
            self.training_listbox.insert(tk.END, display_text)
        
        if not training_data:
            self.training_listbox.insert(tk.END, "No training data yet - add some URLs above!")
    
    def clear_training_data(self):
        """Clear all training data"""
        if messagebox.askyesno("Clear Training Data", 
                             "Are you sure you want to clear all training data? This cannot be undone!"):
            global training_data
            training_data = []
            
            # Save empty training data
            with open(TRAINING_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            self.load_training_history()
            self.training_status.set("🗑️ Training data cleared!")
            messagebox.showinfo("Cleared", "All training data has been cleared!")
    
    def refresh_notes_dropdown(self):
        """Refresh the notes dropdown with current notes"""
        if hasattr(self, 'notes_dropdown'):
            notes_list = list(embeddings_db.keys()) if embeddings_db else ["No notes available"]
            self.notes_dropdown['values'] = notes_list

def main():
    initialize()
    root = tk.Tk()
    app = HrudhiApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()