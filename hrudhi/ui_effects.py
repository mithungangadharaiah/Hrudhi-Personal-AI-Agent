"""
Enhanced UI Components for Hrudhi
Additional visual effects and animations
"""
import tkinter as tk
from tkinter import ttk
import math
import time

class FloatingParticles:
    """Animated floating particles for background effect"""
    def __init__(self, canvas, num_particles=15):
        self.canvas = canvas
        self.particles = []
        
        for _ in range(num_particles):
            x = self.canvas.winfo_reqwidth() * 0.1 + (self.canvas.winfo_reqwidth() * 0.8) * (len(self.particles) % 5) / 4
            y = self.canvas.winfo_reqheight() * 0.2 + (self.canvas.winfo_reqheight() * 0.6) * (len(self.particles) // 5) / 3
            particle = self.canvas.create_oval(x-2, y-2, x+2, y+2, 
                                             fill="#4A90E2", outline="", 
                                             stipple="gray25")
            self.particles.append({
                'id': particle,
                'x': x,
                'y': y,
                'dx': (hash(str(particle)) % 20 - 10) / 100,
                'dy': (hash(str(particle)) % 20 - 10) / 100,
                'alpha': 0.3
            })
        
        self.animate_particles()
    
    def animate_particles(self):
        for particle in self.particles:
            # Update position
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Bounce off edges
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if particle['x'] <= 0 or particle['x'] >= canvas_width:
                particle['dx'] *= -1
            if particle['y'] <= 0 or particle['y'] >= canvas_height:
                particle['dy'] *= -1
            
            # Update canvas position
            self.canvas.coords(particle['id'], 
                             particle['x']-2, particle['y']-2,
                             particle['x']+2, particle['y']+2)
        
        # Continue animation
        self.canvas.after(50, self.animate_particles)

class TypewriterEffect:
    """Typewriter effect for text display"""
    def __init__(self, text_widget, text, delay=50):
        self.text_widget = text_widget
        self.text = text
        self.delay = delay
        self.index = 0
        self.start_typing()
    
    def start_typing(self):
        if self.index < len(self.text):
            self.text_widget.insert(tk.END, self.text[self.index])
            self.index += 1
            self.text_widget.after(self.delay, self.start_typing)

class PulseEffect:
    """Pulse effect for widgets"""
    def __init__(self, widget, min_scale=0.95, max_scale=1.05, speed=0.02):
        self.widget = widget
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.speed = speed
        self.scale = min_scale
        self.direction = 1
        self.original_font = None
        
        # Store original font
        try:
            self.original_font = widget['font']
            if isinstance(self.original_font, str):
                # Parse font string
                parts = self.original_font.split()
                self.font_family = parts[0]
                self.font_size = int(parts[1]) if len(parts) > 1 else 12
            else:
                self.font_family = self.original_font[0] if self.original_font else 'Arial'
                self.font_size = self.original_font[1] if len(self.original_font) > 1 else 12
        except:
            self.font_family = 'Arial'
            self.font_size = 12
        
        self.animate()
    
    def animate(self):
        # Update scale
        self.scale += self.direction * self.speed
        
        # Reverse direction at limits
        if self.scale >= self.max_scale:
            self.direction = -1
        elif self.scale <= self.min_scale:
            self.direction = 1
        
        # Apply scale to font size
        try:
            new_size = int(self.font_size * self.scale)
            self.widget.configure(font=(self.font_family, new_size))
        except:
            pass
        
        # Continue animation
        self.widget.after(20, self.animate)

class GlowEffect:
    """Glow effect for text"""
    def __init__(self, canvas, text, x, y, font=('Arial', 24, 'bold'), glow_color='#4A90E2'):
        self.canvas = canvas
        self.text = text
        self.x = x
        self.y = y
        self.font = font
        self.glow_color = glow_color
        self.glow_intensity = 0
        self.glow_direction = 1
        
        # Create glow layers
        self.glow_layers = []
        for i in range(3):
            layer = self.canvas.create_text(
                x + i, y + i, text=text, font=font,
                fill=glow_color, anchor='center'
            )
            self.glow_layers.append(layer)
        
        # Main text
        self.main_text = self.canvas.create_text(
            x, y, text=text, font=font,
            fill='white', anchor='center'
        )
        
        self.animate_glow()
    
    def animate_glow(self):
        # Update glow intensity
        self.glow_intensity += self.glow_direction * 0.05
        
        if self.glow_intensity >= 1:
            self.glow_direction = -1
        elif self.glow_intensity <= 0.3:
            self.glow_direction = 1
        
        # Apply glow effect
        alpha = int(255 * self.glow_intensity)
        glow_color = f"#{alpha:02x}{alpha//2:02x}{255:02x}"
        
        for layer in self.glow_layers:
            try:
                self.canvas.itemconfig(layer, fill=glow_color)
            except:
                pass
        
        # Continue animation
        self.canvas.after(50, self.animate_glow)

def add_gradient_background(widget, color1='#1E1E2E', color2='#2A2A3A'):
    """Add gradient background effect"""
    # This is a simplified version - full gradients require more complex implementation
    widget.configure(bg=color1)
    return widget

def create_rounded_button(parent, text, command=None, bg_color='#4A90E2', hover_color='#357ABD'):
    """Create a custom rounded button"""
    button_frame = tk.Frame(parent, bg=bg_color, relief='flat', bd=1)
    button_label = tk.Label(button_frame, text=text, bg=bg_color, fg='white',
                           font=('Arial', 12, 'bold'), cursor='hand2')
    button_label.pack(padx=15, pady=8)
    
    def on_enter(e):
        button_frame.configure(bg=hover_color)
        button_label.configure(bg=hover_color)
    
    def on_leave(e):
        button_frame.configure(bg=bg_color)
        button_label.configure(bg=bg_color)
    
    def on_click(e):
        if command:
            command()
    
    button_frame.bind("<Enter>", on_enter)
    button_frame.bind("<Leave>", on_leave)
    button_frame.bind("<Button-1>", on_click)
    button_label.bind("<Enter>", on_enter)
    button_label.bind("<Leave>", on_leave)
    button_label.bind("<Button-1>", on_click)
    
    return button_frame