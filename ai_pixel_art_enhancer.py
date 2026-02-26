import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from collections import defaultdict
import math
import colorsys

class AIPixelArtEnhancer:
    """
    ComfyUI Node for enhancing AI-generated pixel art with advanced processing options
    Now supports batch processing for video frames
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to convert into pixel art."}),
                "pixel_size": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 128, 
                    "step": 1,
                    "display": "number",
                    "tooltip": "Size of each 'grain' in pixels. Output resolution will match input exactly."
                }),
                "conversion_method": (["most_frequent", "average", "neighbor_aware", "brightness_weighted_light", "brightness_weighted_dark", "edge_preserving"], {"tooltip": "Algorithm used for conversion. 'most_frequent' (mode), 'average' (mean), 'neighbor_aware' (contextual), 'brightness_weighted_light' (favors brights), 'brightness_weighted_dark' (favors darks), 'edge_preserving' (keeps structural detail)."}),
                "color_similarity_threshold": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Threshold for color clustering. Higher values group more colors together."
                }),
            },
            "optional": {
                "enable_ai_enhancement": ("BOOLEAN", {"default": True, "tooltip": "Enable AI preprocessing and post-processing for better results."}),
                "noise_reduction": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Strength of noise reduction filter."
                }),
                "edge_enhancement": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Intensity of edge enhancement to preserve structural details."
                }),
                "color_quantization": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of colors in the final output. Lower values give a more retro look."
                }),
                "dithering_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Floyd-Steinberg dithering intensity to smoothly blend colors."
                }),
                "palette_extraction": (["median_cut", "fast_octree", "k_means", "k_means_outlier_preserved", "k_means_vibrant_preserved", "uniform_binning"], {"default": "median_cut", "tooltip": "Method to extract colors. 'median_cut' (fast and balanced), 'fast_octree' (fast for transparency), 'k_means' (accurate 3D spatial), 'k_means_outlier_preserved' (boosts rare high-contrast pixels), 'k_means_vibrant_preserved' (boosts highly saturated/pure colors), 'uniform_binning' (basic grid)."}),
                "color_matching_space": (["rgb", "lab", "oklab"], {"default": "rgb", "tooltip": "Color space to use when mapping pixels to the allowed palette. 'rgb' (standard linear math), 'lab' (CIELAB human perception), 'oklab' (newer smooth perception, prevents hue shifts)."}),
                "contrast_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Contrast adjustment multiplier."
                }),
                "saturation_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Color saturation multiplier."
                }),
                "preserve_details": ("BOOLEAN", {"default": True, "tooltip": "Enable detail preservation during processing."}),
                "anti_aliasing": ("BOOLEAN", {"default": False, "tooltip": "Apply subtle anti-aliasing to final output to blend jagged edges."}),
                "batch_processing": ("BOOLEAN", {"default": True, "tooltip": "Process entire batches of images (e.g., video frames). Disable for single images."}),
                "input_palette": ("IMAGE", {"tooltip": "Provide an image to strictly map the result to its exact colors. Overrides 'color_quantization' maximum colors."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced_image", "comparison_grid", "palette")
    FUNCTION = "enhance_pixel_art"
    CATEGORY = "image/enhancement"
    
    def enhance_pixel_art(self, image, pixel_size, conversion_method, 
                         color_similarity_threshold, enable_ai_enhancement=True,
                         noise_reduction=0.3, edge_enhancement=0.5, color_quantization=16,
                         dithering_strength=0.0, palette_extraction="median_cut", color_matching_space="rgb",
                         contrast_boost=1.0, saturation_boost=1.0,
                         preserve_details=True, anti_aliasing=False, batch_processing=True,
                         input_palette=None):
        
        # Extract custom palette colors if provided
        palette_colors = None
        if input_palette is not None:
            extracted_colors = self.extract_colors_from_tensor(input_palette, color_quantization, method=palette_extraction)
            if len(extracted_colors) > 0:
                palette_colors = extracted_colors
            else:
                print("Warning: Provided input_palette was empty or invalid. Ignoring palette input.")
        
        global_palette = set()
        
        # Handle batch processing
        if len(image.shape) == 4 and batch_processing:
            # Process each frame in the batch
            batch_size = image.shape[0]
            enhanced_frames = []
            comparison_frames = []
            
            print(f"Processing {batch_size} frames...")
            
            for i in range(batch_size):
                frame = image[i]  # Get individual frame
                
                # Process single frame
                enhanced_frame, comparison_frame, frame_colors = self.process_single_frame(
                    frame, pixel_size, conversion_method,
                    color_similarity_threshold, enable_ai_enhancement,
                    noise_reduction, edge_enhancement, color_quantization,
                    dithering_strength, palette_extraction, color_matching_space, 
                    contrast_boost, saturation_boost, preserve_details, anti_aliasing, palette_colors
                )
                
                enhanced_frames.append(enhanced_frame)
                comparison_frames.append(comparison_frame)
                for c in frame_colors:
                    global_palette.add(c)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Processed frame {i + 1}/{batch_size}")
            
            # Stack frames back into batch tensors
            enhanced_batch = torch.stack(enhanced_frames, dim=0)
            comparison_batch = torch.stack(comparison_frames, dim=0)
            palette_tensor = self.create_palette_image(list(global_palette))
            
            return (enhanced_batch, comparison_batch, palette_tensor)
        
        else:
            # Single image processing (legacy behavior)
            if len(image.shape) == 4:
                frame = image[0]  # Take first frame if batch with single frame
            else:
                frame = image
            
            enhanced_frame, comparison_frame, frame_colors = self.process_single_frame(
                frame, pixel_size, conversion_method,
                color_similarity_threshold, enable_ai_enhancement,
                noise_reduction, edge_enhancement, color_quantization,
                dithering_strength, palette_extraction, color_matching_space,
                contrast_boost, saturation_boost, preserve_details, anti_aliasing, palette_colors
            )
            
            for c in frame_colors:
                global_palette.add(c)
                
            # Add batch dimension for consistency
            enhanced_batch = enhanced_frame.unsqueeze(0)
            comparison_batch = comparison_frame.unsqueeze(0)
            palette_tensor = self.create_palette_image(list(global_palette))
            
            return (enhanced_batch, comparison_batch, palette_tensor)
    
    def process_single_frame(self, frame, pixel_size, conversion_method, 
                           color_similarity_threshold, enable_ai_enhancement,
                           noise_reduction, edge_enhancement, color_quantization,
                           dithering_strength, palette_extraction, color_matching_space,
                           contrast_boost, saturation_boost,
                           preserve_details, anti_aliasing, palette_colors=None):
        """Process a single frame/image"""
        
        # Convert from tensor [H, W, C] to numpy array
        img_array = (frame.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        
        # Calculate grid size based on pixel_size
        orig_width, orig_height = pil_image.size
        grid_width = max(1, orig_width // pixel_size)
        grid_height = max(1, orig_height // pixel_size)
        
        # AI Enhancement preprocessing
        if enable_ai_enhancement:
            pil_image = self.ai_preprocess(
                pil_image, 
                noise_reduction, 
                edge_enhancement, 
                contrast_boost, 
                saturation_boost,
                preserve_details
            )
        
        # Convert to pixel art
        pixel_art = self.convert_to_pixel_art(
            pil_image,
            grid_width,
            grid_height,
            conversion_method,
            color_similarity_threshold
        )
        
        frame_colors = []
        # Post-processing enhancements
        if enable_ai_enhancement or palette_colors is not None:
            pixel_art, frame_colors = self.post_process_pixel_art(
                pixel_art,
                color_quantization,
                dithering_strength,
                anti_aliasing,
                palette_colors,
                palette_extraction,
                color_matching_space
            )
        else:
            frame_colors = self.extract_colors_from_pil(pixel_art)
        
        # Scale up to exactly the original dimensions (maintaining the pixel block size)
        final_image = pixel_art.resize(
            (orig_width, orig_height),
            Image.NEAREST
        )
        
        # Create comparison grid
        comparison = self.create_comparison_grid(pil_image, pixel_art, final_image)
        
        # Convert back to tensor format (without batch dimension)
        enhanced_tensor = self.pil_to_tensor_single(final_image)
        comparison_tensor = self.pil_to_tensor_single(comparison)
        
        return (enhanced_tensor, comparison_tensor, frame_colors)
    
    def ai_preprocess(self, image, noise_reduction, edge_enhancement, contrast_boost, saturation_boost, preserve_details):
        """AI-specific preprocessing to improve pixel art conversion quality"""
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Noise reduction using bilateral filter
        if noise_reduction > 0:
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            d = int(9 * noise_reduction)
            sigma_color = 75 * noise_reduction
            sigma_space = 75 * noise_reduction
            img_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
            img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Edge enhancement
        if edge_enhancement > 0 and preserve_details:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
            
            # Blend edges back into the image
            edges_colored = np.stack([edges, edges, edges], axis=2)
            img_array = np.clip(
                img_array.astype(np.float32) + edges_colored.astype(np.float32) * edge_enhancement * 0.3,
                0, 255
            ).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        
        # Contrast and saturation adjustments
        if contrast_boost != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_boost)
        
        if saturation_boost != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_boost)
        
        return image
    
    def convert_to_pixel_art(self, image, grid_width, grid_height, method, similarity_threshold):
        """Convert image to pixel art using various methods"""
        
        # Calculate cell dimensions
        img_width, img_height = image.size
        cell_width = img_width / grid_width
        cell_height = img_height / grid_height
        
        # Create output image
        pixel_art = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))
        pixels = pixel_art.load()
        
        img_array = np.array(image.convert('RGBA'))
        
        for j in range(grid_height):
            for i in range(grid_width):
                # Calculate cell bounds
                x1 = int(i * cell_width)
                y1 = int(j * cell_height)
                x2 = int((i + 1) * cell_width)
                y2 = int((j + 1) * cell_height)
                
                # Extract cell pixels
                cell_pixels = img_array[y1:y2, x1:x2]
                
                # Skip transparent cells
                if cell_pixels.size == 0:
                    continue
                
                # Calculate representative color based on method
                color = self.get_representative_color(
                    cell_pixels, method, similarity_threshold
                )
                
                if color is not None:
                    pixels[i, j] = color
        
        return pixel_art
    
    def get_representative_color(self, cell_pixels, method, similarity_threshold):
        """Calculate representative color for a cell using various methods"""
        
        # Flatten the cell pixels and filter out transparent ones
        flat_pixels = cell_pixels.reshape(-1, 4)
        opaque_pixels = flat_pixels[flat_pixels[:, 3] > 0]  # Alpha > 0
        
        if len(opaque_pixels) == 0:
            return None
        
        rgb_pixels = opaque_pixels[:, :3]  # Drop alpha channel
        
        if method == "average":
            color = np.mean(rgb_pixels, axis=0).astype(int)
            return tuple(color.tolist() + [255])
        
        elif method == "most_frequent":
            return self.get_most_frequent_color(rgb_pixels, similarity_threshold)
        
        elif method == "neighbor_aware":
            # Expand sampling area for better context
            return self.get_neighbor_aware_color(rgb_pixels)
        
        elif method in ["brightness_weighted_light", "brightness_weighted_dark"]:
            return self.get_brightness_weighted_color(rgb_pixels, method, similarity_threshold)
        
        elif method == "edge_preserving":
            return self.get_edge_preserving_color(cell_pixels)
        
        else:
            # Fallback to average
            color = np.mean(rgb_pixels, axis=0).astype(int)
            return tuple(color.tolist() + [255])
    
    def get_most_frequent_color(self, pixels, similarity_threshold):
        """Find most frequent color with clustering using NumPy for safety"""
        if len(pixels) == 0:
            return None
        
        # Convert to numpy array and ensure proper dtype
        pixels = np.array(pixels, dtype=np.float32)[:, :3]  # RGB only
        
        # Simple clustering based on color similarity
        clusters = []
        
        for pixel in pixels:
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            # Find matching cluster
            matched_cluster = None
            min_distance = float('inf')
            
            for cluster in clusters:
                rep = np.array(cluster['representative'], dtype=np.float32)
                current_pixel = np.array([r, g, b], dtype=np.float32)
                
                # Calculate Euclidean distance using NumPy
                distance = np.linalg.norm(current_pixel - rep)
                
                if distance <= similarity_threshold and distance < min_distance:
                    matched_cluster = cluster
                    min_distance = distance
            
            if matched_cluster:
                # Add pixel to existing cluster
                matched_cluster['pixels'].append([r, g, b])
                matched_cluster['count'] += 1
                
                # Update representative as running average using NumPy
                pixels_array = np.array(matched_cluster['pixels'], dtype=np.float32)
                new_representative = np.mean(pixels_array, axis=0)
                
                # Clamp to valid color range
                matched_cluster['representative'] = np.clip(new_representative, 0, 255).astype(int).tolist()
            else:
                # Create new cluster
                clusters.append({
                    'representative': [int(np.clip(r, 0, 255)), 
                                    int(np.clip(g, 0, 255)), 
                                    int(np.clip(b, 0, 255))],
                    'pixels': [[r, g, b]],
                    'count': 1
                })
        
        # Return color from largest cluster
        if clusters:
            largest_cluster = max(clusters, key=lambda x: x['count'])
            rep_color = largest_cluster['representative']
            
            # Ensure final color is valid
            final_color = [int(np.clip(c, 0, 255)) for c in rep_color]
            return tuple(final_color + [255])
        
        return None
        
    def get_neighbor_aware_color(self, pixels):
        """Get color using neighbor-aware sampling"""
        # For now, use weighted average with more weight on central pixels
        if len(pixels) == 0:
            return None
        
        # Simple implementation: return average color
        color = np.mean(pixels, axis=0).astype(int)
        return tuple(color.tolist() + [255])
    
    def get_brightness_weighted_color(self, pixels, method, similarity_threshold):
        """Get color weighted by brightness preference"""
        if len(pixels) == 0:
            return None
        
        weights = []
        for pixel in pixels:
            r, g, b = pixel[:3]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            
            if method == "brightness_weighted_light":
                weight = brightness / 255.0
            else:  # brightness_weighted_dark
                weight = (255 - brightness) / 255.0
            
            weights.append(0.25 + 0.75 * weight)  # Minimum weight of 0.25
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted average
        weighted_color = np.average(pixels, axis=0, weights=weights)
        return tuple(weighted_color.astype(int).tolist() + [255])
    
    def get_edge_preserving_color(self, cell_pixels):
        """Get color while preserving edges"""
        # Apply edge detection within the cell
        if cell_pixels.size == 0:
            return None
        
        gray = np.mean(cell_pixels[:,:,:3], axis=2).astype(np.uint8)
        
        # Simple edge detection
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            edges = cv2.Canny(gray, 50, 150)
            if np.sum(edges) > 0:
                # If edges detected, use median color to preserve structure
                flat_pixels = cell_pixels.reshape(-1, 4)
                opaque_pixels = flat_pixels[flat_pixels[:, 3] > 0][:, :3]
                if len(opaque_pixels) > 0:
                    color = np.median(opaque_pixels, axis=0).astype(int)
                    return tuple(color.tolist() + [255])
        
        # Fallback to average
        flat_pixels = cell_pixels.reshape(-1, 4)
        opaque_pixels = flat_pixels[flat_pixels[:, 3] > 0][:, :3]
        if len(opaque_pixels) > 0:
            color = np.mean(opaque_pixels, axis=0).astype(int)
            return tuple(color.tolist() + [255])
        
        return None
    
    def post_process_pixel_art(self, pixel_art, color_quantization, dithering_strength, anti_aliasing, palette_colors=None, palette_extraction="median_cut", color_matching_space="rgb"):
        """Apply post-processing enhancements"""
        
        # Color quantization / Palette application
        if palette_colors is not None:
            pixel_art = self.apply_palette(pixel_art, palette_colors, color_matching_space)
        elif color_quantization <= 256:
            # When forcing quantization dynamically, we'll still do the matching in the requested space
            pixel_art = self.quantize_colors(pixel_art, color_quantization, palette_extraction)
            if color_matching_space != "rgb":
                # Ensure the quantized image maps strictly using the perceptual space
                extracted_colors = self.extract_colors_from_pil(pixel_art)
                if len(extracted_colors) > 0:
                    pixel_art = self.apply_palette(pixel_art, extracted_colors, color_matching_space)

        frame_colors = self.extract_colors_from_pil(pixel_art)
        
        # Dithering
        if dithering_strength > 0 and palette_colors is None:
            pixel_art = self.apply_dithering(pixel_art, dithering_strength)
        
        # Anti-aliasing (minimal, preserve pixel art aesthetic)
        if anti_aliasing:
            pixel_art = self.apply_subtle_antialiasing(pixel_art)
        
        return pixel_art, frame_colors
    
    def quantize_colors(self, image, num_colors, method="median_cut"):
        """Reduce the number of colors in the image"""
        if method == "uniform_binning":
            return self.manual_quantize_rgba(image, num_colors, force_binning=True)
            
        if method == "k_means":
            return self.manual_quantize_rgba(image, num_colors)
            
        if method == "k_means_outlier_preserved":
            return self.manual_quantize_rgba(image, num_colors, enable_outlier_preservation=True)
            
        if method == "k_means_vibrant_preserved":
            return self.manual_quantize_rgba(image, num_colors, enable_vibrant_preservation=True)
            
        pil_method = Image.MAXCOVERAGE if method == "median_cut" else Image.FASTOCTREE
            
        if image.mode == 'RGBA':
            # Fast Octree works better with transparency mapping, but MAXCOVERAGE works too.
            try:
                if pil_method == Image.MAXCOVERAGE:
                    # Pillow doesn't formally support MAXCOVERAGE with RGBA directly without losing alpha sometimes,
                    # but typically method 2 (FASTOCTREE) is required for full alpha. Let's try what's requested.
                    return image.quantize(colors=num_colors, method=pil_method).convert('RGBA')
                return image.quantize(colors=num_colors, method=pil_method).convert('RGBA')
            except Exception:
                # Fallback: K-Means manual quantization
                return self.manual_quantize_rgba(image, num_colors)
        else:
            # For RGB images, both methods work purely in PIL
            return image.quantize(colors=num_colors, method=pil_method).convert('RGBA')
    
    def manual_quantize_rgba(self, image, num_colors, force_binning=False, enable_outlier_preservation=False, enable_vibrant_preservation=False):
        """Manual color quantization using K-means clustering (forces RGBA)"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        img_array = np.array(image)
        
        # Separate pixels with alpha > 0 (opaque) and alpha = 0 (transparent)
        height, width, channels = img_array.shape
        pixels = img_array.reshape(-1, 4)
        
        # Find opaque pixels
        opaque_mask = pixels[:, 3] > 0
        opaque_pixels = pixels[opaque_mask]
        
        if len(opaque_pixels) == 0:
            return image  # All transparent
        
        # Use only RGB channels for clustering
        rgb_pixels = opaque_pixels[:, :3].astype(np.float32)
        
        # Simple K-means clustering
        if not force_binning:
            try:
                from sklearn.cluster import KMeans
                
                # Outlier Preservation Logic
                train_pixels = rgb_pixels
                if enable_outlier_preservation and len(rgb_pixels) > 0:
                    # Calculate mean color and distance of every pixel to the mean
                    mean_color = np.mean(rgb_pixels, axis=0)
                    distances = np.sum((rgb_pixels - mean_color) ** 2, axis=1)
                    
                    # Target top 5% most outlying contrasting pixels
                    threshold_dist = np.percentile(distances, 95)
                    outliers = rgb_pixels[distances > threshold_dist]
                    
                    # Amplify outlier presence by duplicating them heavily into the training set
                    # This forces KMeans to allocate a cluster center to them
                    if len(outliers) > 0:
                        weight_multiplier = max(1, len(rgb_pixels) // (len(outliers) * 2))
                        repeated_outliers = np.repeat(outliers, weight_multiplier, axis=0)
                        train_pixels = np.vstack([rgb_pixels, repeated_outliers])
                        
                # Vibrant Preservation Logic
                if enable_vibrant_preservation and len(rgb_pixels) > 0:
                    # Calculate the difference between the max and min RGB channels for each pixel
                    # Highly vibrant/pure colors will have a large difference (e.g., Red 255 vs Blue 0)
                    max_channel = np.max(rgb_pixels, axis=1)
                    min_channel = np.min(rgb_pixels, axis=1)
                    vibrancy = max_channel - min_channel
                    
                    # Target top 5% most vibrant pixels
                    threshold_vibrancy = np.percentile(vibrancy, 95)
                    vibrant_pixels = rgb_pixels[vibrancy > threshold_vibrancy]
                    
                    # Amplify vibrant pixel presence
                    if len(vibrant_pixels) > 0:
                        weight_multiplier = max(1, len(rgb_pixels) // (len(vibrant_pixels) * 2))
                        repeated_vibrant = np.repeat(vibrant_pixels, weight_multiplier, axis=0)
                        train_pixels = np.vstack([rgb_pixels, repeated_vibrant])
                
                kmeans = KMeans(n_clusters=min(num_colors, len(rgb_pixels)), 
                              random_state=42, n_init=10)
                
                kmeans.fit(train_pixels)
                cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
                
                # Assign labels to the original (unweighted) pixels
                distances = np.sum((rgb_pixels[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2, axis=2)
                cluster_labels = np.argmin(distances, axis=1)
                
            except ImportError:
                force_binning = True
                print("sklearn not found. Falling back to uniform binning.")

        if force_binning:
            # Fallback: simple uniform binning
            cluster_centers = []
            step = 256 // int(np.cbrt(num_colors))
            for r in range(0, 256, step):
                for g in range(0, 256, step):
                    for b in range(0, 256, step):
                        cluster_centers.append([r, g, b])
                        if len(cluster_centers) >= num_colors:
                            break
                    if len(cluster_centers) >= num_colors:
                        break
                if len(cluster_centers) >= num_colors:
                    break
            
            cluster_centers = np.array(cluster_centers[:num_colors])
            
            # Assign each pixel to nearest cluster center
            cluster_labels = []
            for pixel in rgb_pixels:
                distances = np.sum((cluster_centers - pixel) ** 2, axis=1)
                cluster_labels.append(np.argmin(distances))
            cluster_labels = np.array(cluster_labels)
        
        # Create quantized image
        quantized_pixels = pixels.copy()
        for i, label in enumerate(cluster_labels):
            idx = np.where(opaque_mask)[0][i]
            quantized_pixels[idx, :3] = cluster_centers[label]
        
        quantized_array = quantized_pixels.reshape(height, width, 4).astype(np.uint8)
        return Image.fromarray(quantized_array, 'RGBA')
    
    def apply_dithering(self, image, strength):
        """Apply Floyd-Steinberg dithering"""
        if strength == 0:
            return image
        
        # Simple dithering implementation
        img_array = np.array(image).astype(np.float32)
        height, width = img_array.shape[:2]
        
        for y in range(height - 1):
            for x in range(1, width - 1):
                old_pixel = img_array[y, x, :3].copy()
                new_pixel = np.round(old_pixel / 32) * 32  # Quantize
                img_array[y, x, :3] = new_pixel
                
                quant_error = (old_pixel - new_pixel) * strength
                
                # Distribute error to neighboring pixels
                img_array[y, x + 1, :3] += quant_error * 7/16
                img_array[y + 1, x - 1, :3] += quant_error * 3/16
                img_array[y + 1, x, :3] += quant_error * 5/16
                img_array[y + 1, x + 1, :3] += quant_error * 1/16
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def apply_subtle_antialiasing(self, image):
        """Apply very subtle anti-aliasing while preserving pixel art look"""
        # Use a very light gaussian blur
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def create_comparison_grid(self, original, pixel_art, final):
        """Create a comparison grid showing original, pixel art, and final result"""
        
        # Resize images to same height for comparison
        target_height = 512
        
        orig_aspect = original.width / original.height
        pixel_aspect = pixel_art.width / pixel_art.height
        final_aspect = final.width / final.height
        
        orig_resized = original.resize((int(target_height * orig_aspect), target_height), Image.LANCZOS)
        pixel_resized = pixel_art.resize((int(target_height * pixel_aspect), target_height), Image.NEAREST)
        final_resized = final.resize((int(target_height * final_aspect), target_height), Image.NEAREST)
        
        # Create comparison grid
        total_width = orig_resized.width + pixel_resized.width + final_resized.width
        comparison = Image.new('RGB', (total_width, target_height), (40, 40, 40))
        
        comparison.paste(orig_resized, (0, 0))
        comparison.paste(pixel_resized, (orig_resized.width, 0))
        comparison.paste(final_resized, (orig_resized.width + pixel_resized.width, 0))
        
        return comparison
    
    def pil_to_tensor_single(self, image):
        """Convert PIL Image to tensor format without batch dimension"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)
        
        return tensor
    
    def pil_to_tensor(self, image):
        """Convert PIL Image to ComfyUI tensor format (legacy method with batch dimension)"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor

    def extract_colors_from_tensor(self, tensor, max_colors, method="median_cut"):
        """Extract dominant RGB colors from a tensor using color quantization"""
        if tensor is None:
            return []
            
        try:
            # Convert first frame of palette tensor to PIL Image
            if len(tensor.shape) == 4:
                img_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            elif len(tensor.shape) == 3:
                img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            else:
                print(f"Warning: Invalid palette tensor shape {tensor.shape}.")
                return []
                
            pil_img = Image.fromarray(img_array)
            
            # Quantize to limit the number of colors based on method
            quantized = self.quantize_colors(pil_img, max_colors, method)
            
            # Extract the unique colors from the quantized image
            return self.extract_colors_from_pil(quantized)
            
        except Exception as e:
            print(f"Error extracting colors from palette input: {e}")
            return []

    def extract_colors_from_pil(self, image):
        """Extract unique RGB colors from a PIL Image"""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            pixels = img_array.reshape(-1, img_array.shape[2])
            if img_array.shape[2] == 4:
                pixels = pixels[pixels[:, 3] > 0]
            rgb_pixels = pixels[:, :3]
        else:
            rgb_pixels = img_array.reshape(-1, 1)
            rgb_pixels = np.repeat(rgb_pixels, 3, axis=1)
        unique_colors = np.unique(rgb_pixels, axis=0)
        return [tuple(c) for c in unique_colors]

    def create_palette_image(self, colors, swatch_size=32):
        """Create an image displaying the palette colors in a roughly square grid"""
        if not colors:
            return torch.zeros((1, swatch_size, swatch_size, 3), dtype=torch.float32)

        # Sort colors somewhat (by hue to keep it visually pleasing)
        colors.sort(key=lambda c: colorsys.rgb_to_hsv(c[0]/255.0, c[1]/255.0, c[2]/255.0)[0])

        num_colors = len(colors)
        
        # Calculate grid dimensions to be as square as possible
        cols = math.ceil(math.sqrt(num_colors))
        rows = math.ceil(num_colors / cols)

        width = cols * swatch_size
        height = rows * swatch_size

        palette_img = Image.new('RGB', (width, height), (0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(palette_img)

        for i, color in enumerate(colors):
            x = (i % cols) * swatch_size
            y = (i // cols) * swatch_size
            draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=tuple(color))

        return self.pil_to_tensor_single(palette_img).unsqueeze(0)

    def rgb_to_lab(self, rgb_array):
        """Convert float32 RGB [0-255] to CIELAB space using OpenCV"""
        rgb_img = (rgb_array.reshape(1, -1, 3).astype(np.float32) / 255.0)
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
        return lab_img.reshape(-1, 3).astype(np.float32)

    def rgb_to_oklab(self, rgb_array):
        """Convert float32 RGB [0-255] to OKLAB space"""
        rgb = rgb_array.astype(np.float32) / 255.0
        
        # sRGB to linear sRGB
        mask = rgb > 0.04045
        rgb_lin = np.empty_like(rgb)
        rgb_lin[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
        rgb_lin[~mask] = rgb[~mask] / 12.92
        
        # linear sRGB to LMS
        M1 = np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ], dtype=np.float32)
        
        lms = rgb_lin @ M1.T
        
        # non-linear LMS
        lms_ = np.cbrt(np.maximum(lms, 0))
        
        # LMS to OKLAB
        M2 = np.array([
            [ 0.2104542553,  0.7936177850, -0.0040720468],
            [ 1.9779984951, -2.4285922050,  0.4505937099],
            [ 0.0259040371,  0.7827717662, -0.8086757660]
        ], dtype=np.float32)
        
        oklab = lms_ @ M2.T
        return oklab

    def apply_palette(self, image, palette_colors, color_matching_space="rgb"):
        """Map each pixel to the nearest color in the palette."""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        pixels = img_array.reshape(-1, img_array.shape[2])

        if img_array.shape[2] == 4:
            opaque_mask = pixels[:, 3] > 0
            rgb_pixels = pixels[opaque_mask, :3]
        else:
            opaque_mask = np.ones(len(pixels), dtype=bool)
            rgb_pixels = pixels[:, :3]

        if len(rgb_pixels) == 0:
            return image

        palette_array = np.array(palette_colors, dtype=np.float32)
        quantized_rgb = np.zeros_like(rgb_pixels)
        
        # Convert spaces once for the palette
        if color_matching_space == "oklab":
            palette_space = self.rgb_to_oklab(palette_array)
        elif color_matching_space == "lab":
            palette_space = self.rgb_to_lab(palette_array)
        else:
            palette_space = palette_array

        # Try to use scipy for much faster distance matrix calculation if available
        try:
            from scipy.spatial.distance import cdist
            use_scipy = True
        except ImportError:
            use_scipy = False

        # Process in chunks to save memory
        chunk_size = 10000
        for i in range(0, len(rgb_pixels), chunk_size):
            chunk_rgb = rgb_pixels[i:i+chunk_size].astype(np.float32)
            
            # Convert space for the image chunk
            if color_matching_space == "oklab":
                chunk_space = self.rgb_to_oklab(chunk_rgb)
            elif color_matching_space == "lab":
                chunk_space = self.rgb_to_lab(chunk_rgb)
            else:
                chunk_space = chunk_rgb

            if use_scipy:
                # Highly optimized C backend for distance matrices
                distances = cdist(chunk_space, palette_space, metric='sqeuclidean')
            else:
                # Fallback purely to numpy broadcasting
                distances = np.sum((chunk_space[:, np.newaxis, :] - palette_space[np.newaxis, :, :]) ** 2, axis=2)
                
            nearest_idx = np.argmin(distances, axis=1)
            quantized_rgb[i:i+chunk_size] = palette_array[nearest_idx]

        if img_array.shape[2] == 4:
            pixels[opaque_mask, :3] = quantized_rgb
            result_array = pixels.reshape(height, width, 4).astype(np.uint8)
            return Image.fromarray(result_array, 'RGBA')
        else:
            result_array = quantized_rgb.reshape(height, width, 3).astype(np.uint8)
            return Image.fromarray(result_array, 'RGB')

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AIPixelArtEnhancer": AIPixelArtEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIPixelArtEnhancer": "AI Pixel Art Enhancer"
}