import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


class UniversalMineralPreprocessor:
    """
    Universal Mineral Image Preprocessor

    A robust preprocessing pipeline designed to work with various mineral types:
    - Sulfur (yellow-green)
    - Obsidian (black)
    - Quartz (transparent/white)
    - And others...

    Strategy:
    Instead of relying on specific color values, this preprocessor uses:
    1. Edge detection (works for all objects)
    2. GrabCut algorithm (intelligent segmentation)
    3. Object-background contrast analysis
    """

    def __init__(self, input_dir, output_dir):
        """
        Initialize the preprocessor with input and output directories.

        Args:
            input_dir: Directory containing raw mineral images
            output_dir: Directory where processed images will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def step1_load_image(self, image_path):
        """
        Step 1: Load image from disk

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image in BGR format

        Raises:
            ValueError: If image cannot be loaded
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return image

    def step2_denoise(self, image):
        """
        Step 2: Noise reduction

        Applies bilateral filtering to smooth the image while preserving edges.
        This reduces noise without blurring important structural details.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised

    def step3_enhance_contrast(self, image):
        """
        Step 3: Contrast enhancement

        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
        local contrast. This helps extract details from both bright and dark minerals.

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def step4_segment_universal(self, image):
        """
        Step 4: Universal segmentation - works for ALL colors

        Multi-stage strategy:
        1. Edge detection (Canny)
        2. Morphological operations to close contours
        3. Contour filling to find the largest object
        4. GrabCut for precise segmentation

        This approach works for:
        - Bright minerals (quartz)
        - Dark minerals (obsidian)
        - Colored minerals (sulfur)

        Args:
            image: Input image

        Returns:
            Binary segmentation mask
        """
        h, w = image.shape[:2]

        # === Stage 1: Edge Detection ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur before Canny (reduces noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

        # === Stage 2: Morphology - Close Contours ===
        # Dilation: widen edges (connects broken lines)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        # Closing: fill small holes
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # === Stage 3: Fill Contours ===
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask with largest contour
        rough_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Filter out noise (too small contours)
            min_area = (h * w) * 0.01  # Minimum 1% of image area
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            if valid_contours:
                # Select largest contour
                largest = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(rough_mask, [largest], -1, 255, -1)

        # If no contour found, return empty mask
        if np.sum(rough_mask) == 0:
            print("  ⚠️ Warning: No object detected. Please check the image.")
            return rough_mask

        # === Stage 4: GrabCut - Precise Segmentation ===
        # GrabCut uses rough_mask as initialization hint
        mask_grabcut = rough_mask.copy()

        # Convert mask to GrabCut format
        # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground
        grabcut_mask = np.where(rough_mask == 255, cv2.GC_FGD, cv2.GC_BGD).astype('uint8')

        # Find bounding box of the object
        y_indices, x_indices = np.where(rough_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()

            # Add margin
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            rect = (x1, y1, x2 - x1, y2 - y1)

            # Apply GrabCut algorithm
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(image, grabcut_mask, rect, bgd_model, fgd_model,
                            iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

                # Convert GrabCut result to binary mask
                final_mask = np.where((grabcut_mask == cv2.GC_FGD) |
                                      (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            except:
                # If GrabCut fails, use rough mask
                print("  ⚠️ Warning: GrabCut failed, using basic mask")
                final_mask = rough_mask
        else:
            final_mask = rough_mask

        # === Stage 5: Post-processing ===
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Fill holes
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return final_mask

    def step5_remove_artifacts(self, image, mask):
        """
        Step 5: Artifact removal and edge smoothing

        Applies morphological operations to remove small artifacts and smooth
        the mask edges for a cleaner result.

        Args:
            image: Input image
            mask: Binary segmentation mask

        Returns:
            Cleaned binary mask
        """
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Smooth edges (removes "staircase" effect)
        # Gaussian blur + threshold produces smoother edges
        blurred_mask = cv2.GaussianBlur(filled, (5, 5), 0)
        _, smooth_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        # Additional morphology for robustness
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Select largest region only
        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(smooth_mask)
            cv2.drawContours(clean_mask, [largest], -1, 255, -1)
            return clean_mask

        return smooth_mask

    def step6_extract_mineral(self, image, mask):
        """
        Step 6: Mineral extraction

        Finds the bounding box of the mineral, adds margin, and crops the region.

        Args:
            image: Input image
            mask: Binary segmentation mask

        Returns:
            Tuple of (cropped_image, cropped_mask)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, mask

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add 30px margin
        margin = 30
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Crop region
        cropped_image = image[y:y + h, x:x + w]
        cropped_mask = mask[y:y + h, x:x + w]

        return cropped_image, cropped_mask

    def step7_normalize_size(self, image, mask, target_size=(512, 512)):
        """
        Step 7: Size normalization to 512x512

        Scales the image while maintaining aspect ratio and adds white padding.

        Args:
            image: Input image
            mask: Binary segmentation mask
            target_size: Target dimensions (default: 512x512)

        Returns:
            Tuple of (normalized_image, normalized_mask)
        """
        h, w = image.shape[:2]

        # Calculate scale factor
        scale = min(target_size[0] / h, target_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create white canvas
        canvas = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 255
        canvas_mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

        # Center the image
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
        canvas_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_mask

        return canvas, canvas_mask

    def process_single_image(self, image_path, visualize=False):
        """
        Complete processing pipeline for a single image

        Args:
            image_path: Path to the image file
            visualize: If True, displays visualization of all processing steps

        Returns:
            Tuple of (processed_image, final_mask, original_image)
        """
        print(f"\n{'=' * 60}")
        print(f"Processing: {image_path.name}")
        print(f"{'=' * 60}")

        # Step 1: Load
        print("Step 1: Loading image...")
        original = self.step1_load_image(image_path)
        print(f"  ✓ Size: {original.shape[1]}x{original.shape[0]} px")

        # Step 2: Denoise
        print("Step 2: Noise reduction...")
        denoised = self.step2_denoise(original)
        print("  ✓ Complete")

        # Step 3: Enhance contrast
        print("Step 3: Contrast enhancement...")
        enhanced = self.step3_enhance_contrast(denoised)
        print("  ✓ Complete")

        # Step 4: Segmentation
        print("Step 4: Universal segmentation (edges + GrabCut)...")
        mask = self.step4_segment_universal(enhanced)
        print("  ✓ Complete")

        # Step 5: Clean artifacts
        print("Step 5: Artifact removal...")
        clean_mask = self.step5_remove_artifacts(enhanced, mask)
        print("  ✓ Complete")

        # Step 6: Extract region
        print("Step 6: Region extraction...")
        extracted, extracted_mask = self.step6_extract_mineral(enhanced, clean_mask)
        print(f"  ✓ Region: {extracted.shape[1]}x{extracted.shape[0]} px")

        # Step 7: Normalize size
        print("Step 7: Normalizing to 512x512...")
        final, final_mask = self.step7_normalize_size(extracted, extracted_mask)
        print("  ✓ Complete")

        if visualize:
            self.visualize_pipeline(
                original, denoised, enhanced, mask,
                clean_mask, extracted, final, final_mask
            )

        return final, final_mask, original

    def visualize_pipeline(self, original, denoised, enhanced, mask,
                           clean_mask, extracted, final, final_mask):
        """
        Visualize all processing steps

        Creates a 3x3 grid showing the progression through each step of the pipeline.

        Args:
            original: Original image
            denoised: Denoised image
            enhanced: Contrast-enhanced image
            mask: Raw segmentation mask
            clean_mask: Cleaned segmentation mask
            extracted: Extracted mineral region
            final: Final normalized image
            final_mask: Final normalized mask
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('1. Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(denoised_rgb)
        axes[0, 1].set_title('2. Denoised', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(enhanced_rgb)
        axes[0, 2].set_title('3. Contrast Enhanced', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('4. Raw Mask', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(clean_mask, cmap='gray')
        axes[1, 1].set_title('5. Clean Mask', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(extracted_rgb)
        axes[1, 2].set_title('6. Extracted', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        axes[2, 0].imshow(final_rgb)
        axes[2, 0].set_title('7. Final (512x512)', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(final_mask, cmap='gray')
        axes[2, 1].set_title('Final Mask', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')

        # Overlay visualization
        overlay = final_rgb.copy()
        overlay[final_mask > 0] = overlay[final_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
        axes[2, 2].imshow(overlay.astype(np.uint8))
        axes[2, 2].set_title('Verification Overlay', fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def process_folder(self, visualize_first=True):
        """
        Process all images in the input folder

        Args:
            visualize_first: If True, displays visualization for the first image
        """
        image_files = list(self.input_dir.glob('*.png')) + \
                      list(self.input_dir.glob('*.jpg')) + \
                      list(self.input_dir.glob('*.jpeg'))

        if not image_files:
            print("❌ No images found!")
            return

        mineral_name = self.input_dir.name

        print(f"\n{'#' * 60}")
        print(f"  PREPROCESSING: {mineral_name.upper()}")
        print(f"{'#' * 60}")
        print(f"Found {len(image_files)} images")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}\n")

        for i, img_path in enumerate(image_files, 1):
            try:
                show_viz = (i == 1 and visualize_first)
                processed, mask, _ = self.process_single_image(img_path, visualize=show_viz)

                # Save results
                output_name = f"{img_path.stem}_processed.png"
                mask_name = f"{img_path.stem}_mask.png"

                cv2.imwrite(str(self.output_dir / output_name), processed)
                cv2.imwrite(str(self.output_dir / mask_name), mask)

                print(f"✅ Saved: {output_name}")

            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")

        print(f"\n{'#' * 60}")
        print(f"✅ COMPLETE! Processed images saved to: {self.output_dir}")
        print(f"{'#' * 60}\n")

    def process_all_minerals(self, minerals_list, visualize_first=True):
        """
        Process ALL minerals in batch mode

        Args:
            minerals_list: List of mineral folder names (e.g., ['sulfur', 'obsidian', 'quartz'])
            visualize_first: If True, displays visualization for the first image of each mineral
        """
        base_input = self.input_dir.parent
        base_output = self.output_dir.parent

        print(f"\n{'#' * 70}")
        print(f"  BATCH PREPROCESSING - ALL MINERALS")
        print(f"{'#' * 70}\n")

        for mineral in minerals_list:
            input_path = base_input / mineral
            output_path = base_output / mineral

            if not input_path.exists():
                print(f"⚠️  Skipped {mineral} - directory does not exist")
                continue

            # Process this mineral
            preprocessor = UniversalMineralPreprocessor(input_path, output_path)
            preprocessor.process_folder(visualize_first=visualize_first)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # ========================================
    # Option 1: Process a single mineral
    # ========================================
    print("\n=== Option 1: Single Mineral ===")

    INPUT_FOLDER = "images/minerals_raw/sulfur"
    OUTPUT_FOLDER = "images/minerals_processed/sulfur"

    preprocessor = UniversalMineralPreprocessor(INPUT_FOLDER, OUTPUT_FOLDER)
    preprocessor.process_folder(visualize_first=True)

    # ========================================
    # Option 2: Process all minerals in batch
    # ========================================
    # print("\n=== Option 2: All Minerals ===")
    #
    # # List of minerals to process
    # MINERALS = ['sulfur', 'obsidian', 'quartz', 'amethyst', 'pyrite']
    #
    # # Base paths
    # BASE_INPUT = "images/minerals_raw/sulfur"  # Used as base path
    # BASE_OUTPUT = "images/minerals_processed/sulfur"
    #
    # preprocessor = UniversalMineralPreprocessor(BASE_INPUT, BASE_OUTPUT)
    # preprocessor.process_all_minerals(MINERALS, visualize_first=True)