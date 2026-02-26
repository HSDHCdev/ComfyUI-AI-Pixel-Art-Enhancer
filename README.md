# AI Pixel Art Enhancer for ComfyUI

A powerful ComfyUI custom node that transforms AI-generated images into high-quality pixel art with advanced processing options and intelligent enhancement algorithms.

<img width="256" height="128" alt="demo2" src="https://github.com/user-attachments/assets/1084e266-d493-4744-9f9d-c5c6b4840472" />

## Recommended LoRA:
https://civitai.com/models/1631459/pixel-art-style-illustrious-by-skormino


## Features

- **Multiple Conversion Methods**: Choose from 6 different pixel art conversion algorithms
- **AI-Enhanced Processing**: Intelligent noise reduction, edge enhancement, and detail preservation
- **Advanced Color Management**: Color quantization, similarity clustering, brightness weighting, and strict perfect-palette mapping
- **Post-Processing Effects**: Dithering, anti-aliasing, and contrast/saturation adjustments
- **Comparison Output**: Visual comparison grid showing original, pixel art, and final enhanced result
- **Palette Extraction**: Extracts the precise colors used into a 32x32 unified swatch grid output
- **Flexible Grain Sizing**: Choose exact pixel sizes for your grains while output resolution matches original
- **Seamless Integrability**: No need to worry about strange dimensions, the node maintains input proportions flawlessly

## Installation

### Method 1: Manual Installation
1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-pixel-art-enhancer.git
   ```

3. Install required dependencies:
   ```bash
   cd ~\ComfyUI_windows_portable\python_embeded                                 #Actual path here
   python.exe -m pip install torch numpy Pillow opencv-python scikit-learn      #If you have existing installations, you probably shouldn't reinstall them
   ```

5. Restart ComfyUI

### Dependencies
- `torch`
- `numpy`
- `Pillow (PIL)`
- `opencv-python`
- `scikit-learn` (optional, for advanced color quantization)

## Usage

### Basic Setup
1. Add the "AI Pixel Art Enhancer" node to your ComfyUI workflow
2. Connect an image input to the node
3. Configure the desired settings
4. The node outputs both the enhanced pixel art and a comparison grid

### Input Parameters

#### Required Parameters
- **image**: Input image tensor
- **pixel_size** (1-128): Size of each "grain" in pixels. Output resolution will match input exactly.
- **conversion_method**: Algorithm used for pixel conversion
- **color_similarity_threshold** (5.0-100.0): Threshold for color clustering

#### Optional Enhancement Parameters
- **enable_ai_enhancement** (Boolean): Enable AI preprocessing and post-processing
- **noise_reduction** (0.0-1.0): Strength of noise reduction filter
- **edge_enhancement** (0.0-2.0): Edge enhancement intensity
- **color_quantization** (4-256): Number of colors in final output
- **dithering_strength** (0.0-1.0): Floyd-Steinberg dithering intensity
- **contrast_boost** (0.5-2.0): Contrast adjustment multiplier
- **saturation_boost** (0.0-2.0): Color saturation multiplier
- **preserve_details** (Boolean): Enable detail preservation during processing
- **anti_aliasing** (Boolean): Apply subtle anti-aliasing to final output
- **input_palette** (Image): Provide an image to strictly map the result to its exact colors. Overrides `color_quantization`.

## Conversion Methods

### 1. Most Frequent
Analyzes each grid cell and selects the most common color using intelligent clustering based on the color similarity threshold.

**Best for**: Images with distinct color regions, logos, simple illustrations

### 2. Average
Calculates the mathematical average of all colors in each grid cell.

**Best for**: Smooth gradients, photographic content, general purpose conversion

### 3. Neighbor Aware
Considers neighboring pixels when determining the representative color for better context awareness.

**Best for**: Complex scenes, maintaining spatial relationships

### 4. Brightness Weighted Light
Prioritizes lighter colors within each grid cell, weighted by luminance.

**Best for**: High-key images, light backgrounds, preserving highlights

### 5. Brightness Weighted Dark
Emphasizes darker colors within each grid cell, weighted by luminance.

**Best for**: Low-key images, dark themes, preserving shadows

### 6. Edge Preserving
Uses edge detection to maintain important structural details during conversion.

**Best for**: Images with fine details, architectural content, complex patterns

## Advanced Features

### AI Enhancement Pipeline
When `enable_ai_enhancement` is true, the node applies a sophisticated processing pipeline:

1. **Preprocessing**:
   - Bilateral filtering for noise reduction
   - Canny edge detection and enhancement
   - Contrast and saturation adjustments

2. **Conversion**: Selected algorithm with optimized parameters

3. **Post-processing**:
   - Intelligent color quantization using K-means clustering
   - Floyd-Steinberg dithering
   - Subtle anti-aliasing (optional)

### Color Management
- **Similarity Clustering**: Groups similar colors together based on Euclidean distance in RGB space
- **Brightness Weighting**: Applies perceptual brightness weighting (0.299R + 0.587G + 0.114B)
- **Transparency Support**: Properly handles RGBA images with transparent regions

## Output

The node provides three outputs:
1. **Enhanced Image**: The final pixel art result at the specified scale
2. **Comparison Grid**: Side-by-side comparison of original, intermediate, and final images
3. **Palette**: A 32x32 block grid image showing all the exact colors used in the generated pixel art, sorted by hue.

## Tips and Best Practices

### Grain Size Selection
- **Small grains (1-4)**: High-detail pixel art, maintains more original information, classic modern indie game look
- **Medium grains (8-16)**: Classic retro game aesthetic, good balance of detail and heavily stylized blocks
- **Large grains (32+)**: Extreme abstract pixelation, best for simplistic icons or heavy block-colors

### Method Selection Guide
- **Portraits**: Use "average" or "brightness_weighted_light"
- **Landscapes**: Try "neighbor_aware" or "edge_preserving"
- **Graphics/UI**: Use "most_frequent" with low similarity threshold
- **Artistic images**: Experiment with "brightness_weighted_dark" and dithering

### Performance Optimization
- Larger grain sizes process faster
- Disable AI enhancement for faster processing on simple images
- Use color quantization values appropriate to your target (16-64 colors typical)

## Example Workflows

### Basic Pixel Art Conversion
```
Load Image → AI Pixel Art Enhancer → Save Image
Settings: pixel_size=8, "most_frequent" method
```

### Enhanced Artistic Processing
```
Load Image → AI Pixel Art Enhancer → Save Image
Settings: pixel_size=4, "edge_preserving" method, AI enhancement enabled,
          16 colors, 0.3 dithering
```

## Troubleshooting
### Common Issues
- **Memory errors with large images**: Reduce grid size or disable AI enhancement
- **Colors look washed out**: Increase contrast_boost and saturation_boost
- **Too much noise**: Increase noise_reduction parameter
- **Loss of detail**: Enable preserve_details and try "edge_preserving" method

### Performance Tips
- Process images at reasonable resolutions (512-1024px recommended)
- Use appropriate grid sizes for your target output
- Disable anti_aliasing for pure pixel art aesthetic

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New conversion algorithms
- Performance improvements
- Bug fixes
- Documentation improvements

## License

This project is licensed under the Apache 2.0. - see the LICENSE file for details.

## Changelog

### v1.1.0
- Added `k_means_vibrant_preserved` palette extraction method to preserve high-contrast, pure colors (like bright red eyes) during color quantization.

### v1.0.0
- Initial release with 6 conversion methods


## Acknowledgments

- Claude
- https://github.com/nygaard91
