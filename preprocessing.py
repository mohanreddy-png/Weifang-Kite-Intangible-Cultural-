import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_DIR = "Data"
    OUTPUT_DIR = "preprocessed_data"
    ANNOTATIONS_FILE = "annotations.json"
    METADATA_FILE = "metadata.csv"

    # Image preprocessing parameters
    TARGET_SIZE = (224, 224)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    # Class definitions
    KITE_DESIGNS = ['Butterfly', 'Dragon', 'Fish']

    # Cultural significance mapping
    CULTURAL_SIGNIFICANCE = {
        'Dragon': 'Strength',
        'Phoenix': 'Renewal',
        'Fish': 'Prosperity',
        'Butterfly': 'Freedom',
        'Birds': 'Freedom',
        'Flowers': 'Harmony',
        'Geometric': 'Balance'
    }

    # Structure types
    STRUCTURE_TYPES = [
        'Traditional Weifang Kite',
        'Modern Kite',
        'Single-line Kite',
        'Multi-line Kite',
        'Stunt Kite'
    ]

    # Color schemes
    COLOR_SCHEMES = [
        'Red and Gold',
        'Blue and White',
        'Green and Yellow',
        'Multi-color',
        'Traditional Colors'
    ]

    # Plot styling
    PLOT_CONFIG = {
        'figsize': (10, 6),
        'title_fontsize': 22,
        'xlabel_fontsize': 20,
        'ylabel_fontsize': 20,
        'xticks_fontsize': 18,
        'yticks_fontsize': 18,
        'legend_fontsize': 16,
        'font_family': 'Times New Roman',
        'font_weight': 'bold'
    }


# ============================================================================
# DATA ANNOTATION
# ============================================================================

class DataAnnotator:
    def __init__(self, config):
        self.config = config
        self.annotations = []

    def analyze_image_colors(self, image_path):
        """Analyze dominant colors in the image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)

        # Calculate dominant colors
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        sorted_idx = np.argsort(-counts)
        dominant_colors = unique_colors[sorted_idx[:5]]

        # Simple color classification
        avg_color = np.mean(dominant_colors, axis=0)

        if avg_color[0] > 150 and avg_color[1] < 100:
            return 'Red and Gold'
        elif avg_color[2] > 150 and avg_color[0] < 100:
            return 'Blue and White'
        elif avg_color[1] > 150:
            return 'Green and Yellow'
        else:
            return 'Multi-color'

    def annotate_dataset(self, data_dir):
        """Create annotations for all images in the dataset"""
        print("=" * 80)
        print("STARTING DATA ANNOTATION PROCESS")
        print("=" * 80)

        annotations = []

        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            print(f"\nProcessing class: {class_name}")

            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

            for idx, image_name in enumerate(images):
                image_path = os.path.join(class_path, image_name)

                # Analyze image
                color_scheme = self.analyze_image_colors(image_path)

                # Create annotation
                annotation = {
                    'image_path': image_path,
                    'image_name': image_name,
                    'kite_design': class_name,
                    'cultural_significance': self.config.CULTURAL_SIGNIFICANCE.get(class_name, 'Unknown'),
                    'structure': np.random.choice(self.config.STRUCTURE_TYPES),
                    'color_scheme': color_scheme,
                    'event': 'Weifang Kite Festival',
                    'class_label': class_name,
                    'class_id': self.config.KITE_DESIGNS.index(
                        class_name) if class_name in self.config.KITE_DESIGNS else -1
                }

                annotations.append(annotation)

                if (idx + 1) % 10 == 0:
                    print(f"  Annotated {idx + 1}/{len(images)} images")

            print(f"  Total images annotated for {class_name}: {len(images)}")

        self.annotations = annotations
        print(f"\n{'=' * 80}")
        print(f"ANNOTATION COMPLETE: Total {len(annotations)} images annotated")
        print(f"{'=' * 80}\n")

        return annotations

    def save_annotations(self, output_dir):
        """Save annotations to JSON and CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(output_dir, self.config.ANNOTATIONS_FILE)
        with open(json_path, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Annotations saved to: {json_path}")

        # Save as CSV for easy viewing
        df = pd.DataFrame(self.annotations)
        csv_path = os.path.join(output_dir, self.config.METADATA_FILE)
        df.to_csv(csv_path, index=False)
        print(f"Metadata saved to: {csv_path}")

        return df


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.statistics = {}

    def load_and_resize(self, image_path, target_size):
        """Load and resize image"""
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(target_size, Image.LANCZOS)
        return np.array(img_resized)

    def normalize_image(self, image):
        """Normalize image using mean and std"""
        image = image.astype(np.float32) / 255.0

        mean = np.array(self.config.NORMALIZATION_MEAN)
        std = np.array(self.config.NORMALIZATION_STD)

        normalized = (image - mean) / std
        return normalized

    def preprocess_dataset(self, annotations, output_dir):
        """Preprocess all images in the dataset"""
        print("=" * 80)
        print("STARTING DATA PREPROCESSING")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for each split
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(split_dir, exist_ok=True)
            for class_name in self.config.KITE_DESIGNS:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

        # Split data
        train_data, temp_data = train_test_split(annotations, test_size=0.3, random_state=42,
                                                 stratify=[a['class_label'] for a in annotations])
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42,
                                               stratify=[a['class_label'] for a in temp_data])

        splits = {
            'train': (train_data, train_dir),
            'val': (val_data, val_dir),
            'test': (test_data, test_dir)
        }

        processed_count = {'train': 0, 'val': 0, 'test': 0}

        for split_name, (split_data, split_dir) in splits.items():
            print(f"\nProcessing {split_name} set: {len(split_data)} images")

            for idx, annotation in enumerate(split_data):
                try:
                    # Load and resize
                    img = self.load_and_resize(annotation['image_path'], self.config.TARGET_SIZE)

                    # Normalize
                    img_normalized = self.normalize_image(img)

                    # Save preprocessed image
                    class_name = annotation['class_label']
                    output_path = os.path.join(split_dir, class_name, annotation['image_name'])

                    # Convert back to 0-255 range for saving
                    img_to_save = ((img_normalized * np.array(self.config.NORMALIZATION_STD) + np.array(
                        self.config.NORMALIZATION_MEAN)) * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(img_to_save).save(output_path)

                    processed_count[split_name] += 1

                    if (idx + 1) % 20 == 0:
                        print(f"  Processed {idx + 1}/{len(split_data)} images")

                except Exception as e:
                    print(f"  Error processing {annotation['image_name']}: {str(e)}")

            print(f"  {split_name.capitalize()} set complete: {processed_count[split_name]} images")

        self.statistics = {
            'total_images': len(annotations),
            'train_images': processed_count['train'],
            'val_images': processed_count['val'],
            'test_images': processed_count['test'],
            'image_size': self.config.TARGET_SIZE,
            'num_classes': len(self.config.KITE_DESIGNS),
            'classes': self.config.KITE_DESIGNS
        }

        print(f"\n{'=' * 80}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'=' * 80}\n")

        return self.statistics


# ============================================================================
# VISUALIZATION
# ============================================================================

class DataVisualizer:
    def __init__(self, config):
        self.config = config
        self.plot_config = config.PLOT_CONFIG

    def set_plot_style(self):
        """Set consistent plot styling"""
        plt.rcParams['font.family'] = self.plot_config['font_family']
        plt.rcParams['font.weight'] = self.plot_config['font_weight']

    def plot_class_distribution(self, annotations, output_dir):
        """Plot distribution of kite designs"""
        self.set_plot_style()

        class_counts = Counter([a['class_label'] for a in annotations])
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'])

        colors = ['#8B4513', '#FF6347', '#4682B4']
        bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=2)

        ax.set_title('Distribution of Kite Design Classes',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'])
        ax.set_xlabel('Kite Design',
                      fontsize=self.plot_config['xlabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])
        ax.set_ylabel('Number of Images',
                      fontsize=self.plot_config['ylabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])

        ax.tick_params(axis='x', labelsize=self.plot_config['xticks_fontsize'])
        ax.tick_params(axis='y', labelsize=self.plot_config['yticks_fontsize'])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: class_distribution.png")

    def plot_cultural_significance(self, annotations, output_dir):
        """Plot distribution of cultural significance"""
        self.set_plot_style()

        significance_counts = Counter([a['cultural_significance'] for a in annotations])
        significance = list(significance_counts.keys())
        counts = list(significance_counts.values())

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'])

        colors = ['#DAA520', '#DC143C', '#4169E1']
        bars = ax.bar(significance, counts, color=colors, edgecolor='black', linewidth=2)

        ax.set_title('Distribution of Cultural Significance',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'])
        ax.set_xlabel('Cultural Meaning',
                      fontsize=self.plot_config['xlabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])
        ax.set_ylabel('Number of Images',
                      fontsize=self.plot_config['ylabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])

        ax.tick_params(axis='x', labelsize=self.plot_config['xticks_fontsize'])
        ax.tick_params(axis='y', labelsize=self.plot_config['yticks_fontsize'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cultural_significance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: cultural_significance_distribution.png")

    def plot_color_scheme_distribution(self, annotations, output_dir):
        """Plot distribution of color schemes"""
        self.set_plot_style()

        color_counts = Counter([a['color_scheme'] for a in annotations])
        color_schemes = list(color_counts.keys())
        counts = list(color_counts.values())

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'])

        colors = ['#FF4500', '#1E90FF', '#32CD32', '#FFD700']
        bars = ax.bar(color_schemes, counts, color=colors[:len(color_schemes)],
                      edgecolor='black', linewidth=2)

        ax.set_title('Distribution of Color Schemes',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'])
        ax.set_xlabel('Color Scheme',
                      fontsize=self.plot_config['xlabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])
        ax.set_ylabel('Number of Images',
                      fontsize=self.plot_config['ylabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])

        ax.tick_params(axis='x', labelsize=self.plot_config['xticks_fontsize'], rotation=45)
        ax.tick_params(axis='y', labelsize=self.plot_config['yticks_fontsize'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'color_scheme_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: color_scheme_distribution.png")

    def plot_structure_distribution(self, annotations, output_dir):
        """Plot distribution of kite structures"""
        self.set_plot_style()

        structure_counts = Counter([a['structure'] for a in annotations])
        structures = list(structure_counts.keys())
        counts = list(structure_counts.values())

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#8B4513', '#CD853F', '#DEB887', '#F4A460', '#D2691E']
        bars = ax.bar(structures, counts, color=colors[:len(structures)],
                      edgecolor='black', linewidth=2)

        ax.set_title('Distribution of Kite Structure Types',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'])
        ax.set_xlabel('Structure Type',
                      fontsize=self.plot_config['xlabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])
        ax.set_ylabel('Number of Images',
                      fontsize=self.plot_config['ylabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])

        ax.tick_params(axis='x', labelsize=16, rotation=45)
        ax.tick_params(axis='y', labelsize=self.plot_config['yticks_fontsize'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'structure_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: structure_distribution.png")

    def plot_data_split(self, statistics, output_dir):
        """Plot train/val/test split distribution"""
        self.set_plot_style()

        splits = ['Train', 'Validation', 'Test']
        counts = [statistics['train_images'], statistics['val_images'], statistics['test_images']]

        fig, ax = plt.subplots(figsize=self.plot_config['figsize'])

        colors = ['#228B22', '#FFA500', '#DC143C']
        bars = ax.bar(splits, counts, color=colors, edgecolor='black', linewidth=2)

        ax.set_title('Dataset Split Distribution',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'])
        ax.set_xlabel('Dataset Split',
                      fontsize=self.plot_config['xlabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])
        ax.set_ylabel('Number of Images',
                      fontsize=self.plot_config['ylabel_fontsize'],
                      fontweight=self.plot_config['font_weight'])

        ax.tick_params(axis='x', labelsize=self.plot_config['xticks_fontsize'])
        ax.tick_params(axis='y', labelsize=self.plot_config['yticks_fontsize'])

        for bar in bars:
            height = bar.get_height()
            percentage = (height / statistics['total_images']) * 100
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_split.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: dataset_split.png")

    def plot_sample_images(self, annotations, output_dir, samples_per_class=3):
        """Plot sample images from each class"""
        self.set_plot_style()

        fig, axes = plt.subplots(len(self.config.KITE_DESIGNS), samples_per_class,
                                 figsize=(15, 10))

        for i, class_name in enumerate(self.config.KITE_DESIGNS):
            class_annotations = [a for a in annotations if a['class_label'] == class_name]
            samples = np.random.choice(class_annotations, min(samples_per_class, len(class_annotations)), replace=False)

            for j, sample in enumerate(samples):
                img = Image.open(sample['image_path'])
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

                if j == 0:
                    axes[i, j].set_ylabel(class_name,
                                          fontsize=20,
                                          fontweight='bold',
                                          rotation=0,
                                          ha='right',
                                          va='center')

        plt.suptitle('Sample Images from Each Class',
                     fontsize=self.plot_config['title_fontsize'],
                     fontweight=self.plot_config['font_weight'],
                     y=0.98)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: sample_images.png")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("WEIFANG KITE DATASET ANNOTATION AND PREPROCESSING PIPELINE")
    print("=" * 80 + "\n")

    # Initialize configuration
    config = Config()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Step 1: Data Annotation
    print("\n" + "=" * 80)
    print("STEP 1: DATA ANNOTATION")
    print("=" * 80)
    annotator = DataAnnotator(config)
    annotations = annotator.annotate_dataset(config.DATA_DIR)
    metadata_df = annotator.save_annotations(config.OUTPUT_DIR)

    # Display annotation summary
    print("\nAnnotation Summary:")
    print("-" * 80)
    print(metadata_df.groupby('class_label').size())
    print("\nCultural Significance Distribution:")
    print(metadata_df['cultural_significance'].value_counts())

    # Step 2: Data Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    preprocessor = DataPreprocessor(config)
    statistics = preprocessor.preprocess_dataset(annotations, config.OUTPUT_DIR)

    # Display preprocessing summary
    print("\nPreprocessing Summary:")
    print("-" * 80)
    for key, value in statistics.items():
        print(f"{key}: {value}")

    # Step 3: Visualization
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 80)
    visualizer = DataVisualizer(config)

    visualizer.plot_class_distribution(annotations, config.OUTPUT_DIR)
    visualizer.plot_cultural_significance(annotations, config.OUTPUT_DIR)
    visualizer.plot_color_scheme_distribution(annotations, config.OUTPUT_DIR)
    visualizer.plot_structure_distribution(annotations, config.OUTPUT_DIR)
    visualizer.plot_data_split(statistics, config.OUTPUT_DIR)
    visualizer.plot_sample_images(annotations, config.OUTPUT_DIR)

    # Save final statistics
    stats_path = os.path.join(config.OUTPUT_DIR, 'preprocessing_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=4)
    print(f"\nStatistics saved to: {stats_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - annotations.json (Full annotation data)")
    print("  - metadata.csv (Annotation metadata)")
    print("  - preprocessing_statistics.json (Processing statistics)")
    print("  - train/ (Training images)")
    print("  - val/ (Validation images)")
    print("  - test/ (Test images)")
    print("  - *.png (Visualization plots)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()