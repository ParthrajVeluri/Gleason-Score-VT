"""
VGG-16 Intensive Training for Prostate Cancer Grading (SICAPv2 Dataset)
Full fine-tuning with aggressive augmentation to maximize GPU utilization

Dataset: SICAPv2 - Prostate Histology Patches

"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math


BASE_DIR = "/workspace/SICAPv2"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TRAIN_EXCEL = os.path.join(BASE_DIR, "partition/Test/Train.xlsx")
TEST_EXCEL = os.path.join(BASE_DIR, "partition/Test/Test.xlsx")

IMG_SIZE = 224
NUM_CLASSES = 4
BATCH_SIZE = 64  
INITIAL_LR = 1e-4
MIN_LR = 1e-7
EPOCHS = 100  
EARLY_STOPPING_PATIENCE = 15
WARMUP_EPOCHS = 5
DEBUG_MODE = False


FINE_TUNE_FROM = 'block1'  


CLASS_NAMES = ['NC (Non-Cancerous)', 'G3 (Grade 3)', 'G4 (Grade 4)', 'G5 (Grade 5)']
CLASS_LABELS = ['NC', 'G3', 'G4', 'G5']


OUTPUT_DIR = "/workspace/results_intensive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_augmentation_layer():
    """Create a Keras augmentation pipeline for histopathology images."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5),  # Full rotation range for histology
        layers.RandomZoom((-0.2, 0.2)),  # Zoom in/out
        layers.RandomTranslation(0.1, 0.1),  # Shift
        layers.RandomContrast(0.2),  # Contrast variation
        layers.RandomBrightness(0.2),  # Brightness variation
    ], name='augmentation')


@tf.function
def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Apply color jittering to simulate stain variations in histopathology."""
    image = tf.image.random_brightness(image, brightness)
    image = tf.image.random_contrast(image, 1-contrast, 1+contrast)
    image = tf.image.random_saturation(image, 1-saturation, 1+saturation)
    image = tf.image.random_hue(image, hue)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


@tf.function
def mixup(images, labels, alpha=0.2):
    """MixUp augmentation - blend images and labels."""
    batch_size = tf.shape(images)[0]
    

    lam = tf.random.uniform([], 0, alpha)
    
    
    indices = tf.random.shuffle(tf.range(batch_size))
    
    
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels

def load_dataframe(excel_path, debug_mode=False):
    """Load and preprocess the Excel file containing image paths and labels."""
    df = pd.read_excel(excel_path)
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(IMAGES_DIR, x))
    
    label_columns = ['NC', 'G3', 'G4', 'G5']
    df['label'] = df[label_columns].values.tolist()
    df['label_int'] = df[label_columns].values.argmax(axis=1)
    
    if debug_mode:
        print(f"[DEBUG MODE] Slicing to first 100 samples from {excel_path}")
        df = df.head(100)
    
    existing = df['image_path'].apply(os.path.exists).sum()
    print(f"Loaded {len(df)} samples from {os.path.basename(excel_path)}, {existing} images found")
    
    return df


def parse_image(image_path, label):
    """Load and preprocess a single image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment_image_intensive(image, label):
    """Apply intensive augmentation for training."""

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    

    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    

    image = color_jitter(image, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
    

    crop_size = tf.random.uniform([], int(IMG_SIZE * 0.8), IMG_SIZE, dtype=tf.int32)
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    return image, label


def create_train_dataset(df, batch_size, use_mixup=True):
    """Create training dataset with intensive augmentation."""
    image_paths = df['image_path'].values
    labels = np.array(df['label'].tolist(), dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment_image_intensive, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)  # Drop remainder for MixUp
    

    if use_mixup:
        dataset = dataset.map(lambda x, y: mixup(x, y, alpha=0.2), 
                             num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_test_dataset(df, batch_size):
    """Create test dataset without augmentation."""
    image_paths = df['image_path'].values
    labels = np.array(df['label'].tolist(), dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_vgg16_intensive(input_shape=(224, 224, 3), num_classes=4):
    """
    Build VGG-16 with full fine-tuning and enhanced classifier head.
    All layers trainable with layer-wise learning rate decay applied via optimizer.
    """
    
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    
    base_model.trainable = True
    
    print(f"Base model: Full fine-tuning ({len(base_model.layers)} layers)")
    print(f"  All {len(base_model.trainable_variables)} weight tensors are trainable")
    
    
    x = base_model.output
    
    
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
    x = layers.Concatenate(name='pool_concat')([gap, gmp])  # 1024 features
    
    
    x = layers.Dense(1024, kernel_initializer='he_normal', name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Activation('relu', name='fc1_relu')(x)
    x = layers.Dropout(0.5, name='fc1_dropout')(x)
    
    x = layers.Dense(512, kernel_initializer='he_normal', name='fc2')(x)
    x = layers.BatchNormalization(name='fc2_bn')(x)
    x = layers.Activation('relu', name='fc2_relu')(x)
    x = layers.Dropout(0.4, name='fc2_dropout')(x)
    
    x = layers.Dense(256, kernel_initializer='he_normal', name='fc3')(x)
    x = layers.BatchNormalization(name='fc3_bn')(x)
    x = layers.Activation('relu', name='fc3_relu')(x)
    x = layers.Dropout(0.3, name='fc3_dropout')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='VGG16_Intensive')
    
    return model


def compute_class_weights(df, smoothing=0.1):
    """Compute smoothed class weights."""
    label_columns = ['NC', 'G3', 'G4', 'G5']
    class_counts = np.array([df[col].sum() for col in label_columns], dtype=np.float32)
    total_samples = class_counts.sum()
    n_classes = len(class_counts)
    
    
    class_weights = {}
    for i, count in enumerate(class_counts):
        weight = total_samples / (n_classes * count)
    
        weight = (1 - smoothing) * weight + smoothing * 1.0
        class_weights[i] = weight
    
    return class_weights


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine Decay learning rate schedule."""
    
    def __init__(self, initial_lr, warmup_steps, decay_steps, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        
        warmup_lr = self.initial_lr * (step / warmup_steps)
        
        
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_step / decay_steps))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        
        lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        
        return lr
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr
        }


class LearningRateLogger(Callback):
    """Log learning rate at end of each epoch."""
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, '__call__'):

            current_lr = float(lr(self.model.optimizer.iterations))
        else:
            current_lr = float(lr)
        print(f"  Learning rate: {current_lr:.2e}")

def predict_with_tta(model, dataset, n_augmentations=5):
    """
    Test-Time Augmentation: Average predictions over augmented versions.
    """
    print(f"Running TTA with {n_augmentations} augmentations...")
    
    all_predictions = []
    
    for i in range(n_augmentations):
        aug_predictions = []
        
        for images, _ in dataset:

            if i > 0:  
                images = tf.image.random_flip_left_right(images)
                images = tf.image.random_flip_up_down(images)
                k = tf.random.uniform([], 0, 4, dtype=tf.int32)
                images = tf.map_fn(lambda x: tf.image.rot90(x, k), images)
            
            preds = model.predict(images, verbose=0)
            aug_predictions.append(preds)
        
        all_predictions.append(np.vstack(aug_predictions))
        print(f"  TTA pass {i+1}/{n_augmentations} complete")
    

    final_predictions = np.mean(all_predictions, axis=0)
    
    return final_predictions


def evaluate_model(model, test_dataset, df_test, use_tta=True):
    """Comprehensive evaluation with optional TTA."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    y_true = df_test['label_int'].values
    
    if use_tta:
        y_pred_probs = predict_with_tta(model, test_dataset, n_augmentations=5)
    else:
        print("Generating predictions...")
        y_pred_probs = model.predict(test_dataset, verbose=1)
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    y_pred_probs = y_pred_probs[:min_len]
    
    results = {}
    

    accuracy = accuracy_score(y_true, y_pred)
    results['accuracy'] = accuracy
    print(f"\n1. Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0
    )
    
    print("\n2. Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    for i, cls_name in enumerate(CLASS_NAMES):
        print(f"{cls_name:<25} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    results['precision_weighted'] = precision_weighted
    results['recall_weighted'] = recall_weighted
    results['f1_weighted'] = f1_weighted
    
    print("-" * 60)
    print(f"{'Weighted Average':<25} {precision_weighted:>10.4f} {recall_weighted:>10.4f} {f1_weighted:>10.4f}")
    

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    results['quadratic_weighted_kappa'] = qwk
    print(f"\n3. Quadratic Weighted Kappa: {qwk:.4f}")
    print("   (Interpretation: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("                    0.61-0.80 substantial, 0.81-1.00 almost perfect)")
    

    print("\n4. One-vs-Rest ROC AUC:")
    try:
        y_true_onehot = np.zeros((len(y_true), NUM_CLASSES))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        for i in range(NUM_CLASSES):
            if len(np.unique(y_true_onehot[:, i])) > 1:
                auc = roc_auc_score(y_true_onehot[:, i], y_pred_probs[:, i])
                print(f"   {CLASS_NAMES[i]}: {auc:.4f}")
        
        auc_macro = roc_auc_score(y_true_onehot, y_pred_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_true_onehot, y_pred_probs, multi_class='ovr', average='weighted')
        results['roc_auc_macro'] = auc_macro
        results['roc_auc_weighted'] = auc_weighted
        print(f"   Macro Average: {auc_macro:.4f}")
        print(f"   Weighted Average: {auc_weighted:.4f}")
    except Exception as e:
        print(f"   Error computing ROC AUC: {e}")
    

    print("\n5. Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    results['confusion_matrix'] = cm
    
    print(f"\n{'':>15}", end='')
    for label in CLASS_LABELS:
        print(f"{label:>8}", end='')
    print("\n" + "-" * 50)
    for i, label in enumerate(CLASS_LABELS):
        print(f"{'Actual ' + label:>15}", end='')
        for j in range(NUM_CLASSES):
            print(f"{cm[i, j]:>8}", end='')
        print()
    print("\n6. Detailed Classification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], 
                                target_names=CLASS_LABELS, digits=4, zero_division=0))
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - VGG-16 Intensive on SICAPv2', fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    return results


def plot_training_history(history):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(history_path, dpi=150)
    plt.close()
    print(f"Training history saved to: {history_path}")



def main():
    print("="*70)
    print("VGG-16 INTENSIVE TRAINING FOR PROSTATE CANCER GRADING (SICAPv2)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    
    epochs = EPOCHS
    if DEBUG_MODE:
        print("\n*** DEBUG MODE ENABLED ***")
        epochs = 3
    
    print("\n" + "-"*50)
    print("Step 1: Loading Data")
    print("-"*50)
    
    df_train = load_dataframe(TRAIN_EXCEL, debug_mode=DEBUG_MODE)
    df_test = load_dataframe(TEST_EXCEL, debug_mode=DEBUG_MODE)
    
    print(f"\nTraining samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    
    print("\nTraining Class Distribution:")
    for i, col in enumerate(CLASS_LABELS):
        count = df_train[col].sum()
        pct = count / len(df_train) * 100
        print(f"  {CLASS_NAMES[i]}: {count} ({pct:.1f}%)")
    print("\n" + "-"*50)
    print("Step 2: Creating Data Pipelines (Intensive Augmentation)")
    print("-"*50)
    
    train_dataset = create_train_dataset(df_train, BATCH_SIZE, use_mixup=True)
    test_dataset = create_test_dataset(df_test, BATCH_SIZE)
    
    train_steps = len(df_train) // BATCH_SIZE
    test_steps = len(df_test) // BATCH_SIZE + (1 if len(df_test) % BATCH_SIZE else 0)
    
    print(f"Train batches per epoch: {train_steps}")
    print(f"Test batches: {test_steps}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Augmentations: Flip, Rotation, Zoom, Color Jitter, Random Crop, MixUp")
    
    print("\n" + "-"*50)
    print("Step 3: Building VGG-16 Model (Full Fine-Tuning)")
    print("-"*50)
    
    model = build_vgg16_intensive(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    
    
    total_steps = train_steps * epochs
    warmup_steps = train_steps * WARMUP_EPOCHS
    
    lr_schedule = WarmupCosineDecay(
        initial_lr=INITIAL_LR,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        min_lr=MIN_LR
    )
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4  
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("\n" + "-"*50)
    print("Step 3b: Computing Class Weights")
    print("-"*50)
    
    class_weights = compute_class_weights(df_train, smoothing=0.1)
    print("Class weights (smoothed):")
    for i, (cls_name, weight) in enumerate(zip(CLASS_NAMES, class_weights.values())):
        count = df_train[CLASS_LABELS[i]].sum()
        print(f"  {cls_name}: {weight:.3f} (n={count})")
    
    print("\n" + "-"*50)
    print("Step 4: Setting Up Callbacks")
    print("-"*50)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        LearningRateLogger()
    ]
    
    print(f"  - EarlyStopping (patience={EARLY_STOPPING_PATIENCE})")
    print(f"  - ModelCheckpoint (save best)")
    print(f"  - Warmup + Cosine Decay LR Schedule")
    print(f"    Initial LR: {INITIAL_LR}")
    print(f"    Warmup epochs: {WARMUP_EPOCHS}")
    print(f"    Min LR: {MIN_LR}")
    
    print("\n" + "-"*50)
    print("Step 5: Training Model")
    print("-"*50)
    print(f"Max Epochs: {epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Expected time: 20-30+ minutes on RTX 5090")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    plot_training_history(history)
    
    print("\n" + "-"*50)
    print("Step 6: Evaluating Model with Test-Time Augmentation")
    print("-"*50)
    
    results = evaluate_model(model, test_dataset, df_test, use_tta=True)
    
    print("\n" + "-"*50)
    print("Comparison: Evaluation WITHOUT TTA")
    print("-"*50)
    results_no_tta = evaluate_model(model, test_dataset, df_test, use_tta=False)
    
    print("\n" + "-"*50)
    print("Step 7: Saving Results")
    print("-"*50)
    
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("VGG-16 INTENSIVE TRAINING RESULTS - SICAPv2 DATASET\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Time: {training_time/60:.1f} minutes\n")
        f.write(f"Training Samples: {len(df_train)}\n")
        f.write(f"Test Samples: {len(df_test)}\n")
        f.write(f"Epochs Trained: {len(history.history['loss'])}\n\n")
        f.write("METRICS (with TTA):\n")
        f.write("-"*30 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {results['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall: {results['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score: {results['f1_weighted']:.4f}\n")
        f.write(f"Quadratic Weighted Kappa: {results['quadratic_weighted_kappa']:.4f}\n")
        if 'roc_auc_macro' in results:
            f.write(f"ROC AUC (Macro): {results['roc_auc_macro']:.4f}\n")
        f.write("\nCONFUSION MATRIX:\n")
        f.write(str(results['confusion_matrix']))
    
    print(f"Results summary saved to: {summary_path}")
    
    model_path = os.path.join(OUTPUT_DIR, 'final_model.h5')
    model.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nKey Results (with TTA):")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa: {results['quadratic_weighted_kappa']:.4f}")
    print(f"  Weighted F1: {results['f1_weighted']:.4f}")
    print(f"\nTraining Time: {training_time/60:.1f} minutes")
    
    return model, history, results


if __name__ == "__main__":
    main()






