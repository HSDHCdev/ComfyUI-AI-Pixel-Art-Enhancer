import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from collections import defaultdict
import math

class AIPixelArtEnhancer:
    """
    ComfyUI Node for enhancing AI-generated pixel art with advanced processing options
    Now supports batch processing for video frames
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_width": ("INT", {
                    "default": 32, 
                    "min": 8, 
                    "max": 256, 
                    "step": 1,
                    "display": "number"
                }),
                "grid_height": ("INT", {
                    "default": 32, 
                    "min": 8, 
                    "max": 256, 
                    "step": 1,
                    "display": "number"
                }),
                "conversion_method": (["most_frequent", "average", "neighbor_aware", "brightness_weighted_light", "brightness_weighted_dark", "edge_preserving"],),
                "color_similarity_threshold": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "output_scale": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "enable_ai_enhancement": ("BOOLEAN", {"default": True}),
                "noise_reduction": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "edge_enhancement": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "color_quantization": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "dithering_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "contrast_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "saturation_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "preserve_details": ("BOOLEAN", {"default": True}),
                "anti_aliasing": ("BOOLEAN", {"default": False}),
                "batch_processing": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced_image", "comparison_grid")
    FUNCTION = "enhance_pixel_art"
    CATEGORY = "image/enhancement"
    
    def enhance_pixel_art(self, image, grid_width, grid_height, conversion_method, 
                         color_similarity_threshold, output_scale, enable_ai_enhancement=True,
                         noise_reduction=0.3, edge_enhancement=0.5, color_quantization=16,
                         dithering_strength=0.0, contrast_boost=1.0, saturation_boost=1.0,
                         preserve_details=True, anti_aliasing=False, batch_processing=True):
        
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
                enhanced_frame, comparison_frame = self.process_single_frame(
                    frame, grid_width, grid_height, conversion_method,
                    color_similarity_threshold, output_scale, enable_ai_enhancement,
                    noise_reduction, edge_enhancement, color_quantization,
                    dithering_strength, contrast_boost, saturation_boost,
                    preserve_details, anti_aliasing
                )
                
                enhanced_frames.append(enhanced_frame)
                comparison_frames.append(comparison_frame)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Processed frame {i + 1}/{batch_size}")
            
            # Stack frames back into batch tensors
            enhanced_batch = torch.stack(enhanced_frames, dim=0)
            comparison_batch = torch.stack(comparison_frames, dim=0)
            
            return (enhanced_batch, comparison_batch)
        
        else:
            # Single image processing (legacy behavior)
            if len(image.shape) == 4:
                frame = image[0]  # Take first frame if batch with single frame
            else:
                frame = image
            
            enhanced_frame, comparison_frame = self.process_single_frame(
                frame, grid_width, grid_height, conversion_method,
                color_similarity_threshold, output_scale, enable_ai_enhancement,
                noise_reduction, edge_enhancement, color_quantization,
                dithering_strength, contrast_boost, saturation_boost,
                preserve_details, anti_aliasing
            )
            
            # Add batch dimension for consistency
            enhanced_batch = enhanced_frame.unsqueeze(0)
            comparison_batch = comparison_frame.unsqueeze(0)
            
            return (enhanced_batch, comparison_batch)
    
    def process_single_frame(self, frame, grid_width, grid_height, conversion_method, 
                           color_similarity_threshold, output_scale, enable_ai_enhancement,
                           noise_reduction, edge_enhancement, color_quantization,
                           dithering_strength, contrast_boost, saturation_boost,
                           preserve_details, anti_aliasing):
        """Process a single frame/image"""
        
        # Convert from tensor [H, W, C] to numpy array
        img_array = (frame.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        
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
        
        # Post-processing enhancements
        if enable_ai_enhancement:
            pixel_art = self.post_process_pixel_art(
                pixel_art,
                color_quantization,
                dithering_strength,
                anti_aliasing
            )
        
        # Scale up the result
        final_image = pixel_art.resize(
            (grid_width * output_scale, grid_height * output_scale),
            Image.NEAREST
        )
        
        # Create comparison grid
        comparison = self.create_comparison_grid(pil_image, pixel_art, final_image)
        
        # Convert back to tensor format (without batch dimension)
        enhanced_tensor = self.pil_to_tensor_single(final_image)
        comparison_tensor = self.pil_to_tensor_single(comparison)
        
        return (enhanced_tensor, comparison_tensor)
    
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
        """Find most frequent color with clustering"""
        if len(pixels) == 0:
            return None
        
        # Simple clustering based on color similarity
        clusters = []
        
        for pixel in pixels:
            r, g, b = pixel[:3]
            
            # Find matching cluster
            matched_cluster = None
            for cluster in clusters:
                rep_r, rep_g, rep_b = cluster['representative'][:3]
                distance = math.sqrt((r-rep_r)**2 + (g-rep_g)**2 + (b-rep_b)**2)
                
                if distance <= similarity_threshold:
                    matched_cluster = cluster
                    break
            
            if matched_cluster:
                matched_cluster['pixels'].append(pixel)
                matched_cluster['count'] += 1
                # Update representative as weighted average
                total = matched_cluster['count']
                old_rep = matched_cluster['representative']
                matched_cluster['representative'] = [
                    int((old_rep[0] * (total-1) + r) / total),
                    int((old_rep[1] * (total-1) + g) / total),
                    int((old_rep[2] * (total-1) + b) / total)
                ]
            else:
                clusters.append({
                    'representative': [int(r), int(g), int(b)],
                    'pixels': [pixel],
                    'count': 1
                })
        
        # Return color from largest cluster
        if clusters:
            largest_cluster = max(clusters, key=lambda x: x['count'])
            return tuple(largest_cluster['representative'] + [255])
        
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
    
    def post_process_pixel_art(self, pixel_art, color_quantization, dithering_strength, anti_aliasing):
        """Apply post-processing enhancements"""
        
        # Color quantization
        if color_quantization < 256:
            pixel_art = self.quantize_colors(pixel_art, color_quantization)
        
        # Dithering
        if dithering_strength > 0:
            pixel_art = self.apply_dithering(pixel_art, dithering_strength)
        
        # Anti-aliasing (minimal, preserve pixel art aesthetic)
        if anti_aliasing:
            pixel_art = self.apply_subtle_antialiasing(pixel_art)
        
        return pixel_art
    
    def quantize_colors(self, image, num_colors):
        """Reduce the number of colors in the image"""
        if image.mode == 'RGBA':
            # For RGBA images, use method 2 (Fast Octree) which supports transparency
            try:
                return image.quantize(colors=num_colors, method=2).convert('RGBA')
            except Exception:
                # Fallback: manual quantization for RGBA
                return self.manual_quantize_rgba(image, num_colors)
        else:
            # For RGB images, any method works
            return image.quantize(colors=num_colors, method=Image.MAXCOVERAGE).convert('RGBA')
    
    def manual_quantize_rgba(self, image, num_colors):
        """Manual color quantization for RGBA images using K-means clustering"""
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
        
        # Simple K-means clustering (you could use sklearn if available)
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(num_colors, len(rgb_pixels)), 
                          random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(rgb_pixels)
            cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
        except ImportError:
            # Fallback: simple binning if sklearn not available
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

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AIPixelArtEnhancer": AIPixelArtEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIPixelArtEnhancer": "AI Pixel Art Enhancer"
}