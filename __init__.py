"""
AI Pixel Art Enhancer - ComfyUI Custom Node
============================================

A comprehensive pixel art enhancement node for ComfyUI that converts regular images 
and AI-generated content into high-quality pixel art with advanced processing options.

Features:
- Multiple conversion algorithms (most frequent, average, brightness-weighted, etc.)
- AI-specific preprocessing (noise reduction, edge enhancement)
- Post-processing effects (dithering, color quantization, anti-aliasing)
- Flexible grid dimensions and output scaling
- Comparison grid generation

Author: Custom Node
Version: 1.0.0
"""

from .ai_pixel_art_enhancer import AIPixelArtEnhancer

# Export the node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AIPixelArtEnhancer": AIPixelArtEnhancer
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIPixelArtEnhancer": "🎨 AI Pixel Art Enhancer"
}

# Optional: Add version info
__version__ = "1.0.0"

# Optional: Add module metadata
__all__ = [
    "AIPixelArtEnhancer",
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS"
]
