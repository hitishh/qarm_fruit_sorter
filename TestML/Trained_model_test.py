import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array
import matplotlib.pyplot as plt

# === Paths ===
image_dir = Path(r"C:\Users\hitis\Desktop\Applied Robotics\Report\Codes\TestML\unripe_T")
model_path = r"C:\Users\hitis\Desktop\Applied Robotics\Report\Codes\Fully Auto\best_model.keras"
output_excel = Path(r"C:\Users\hitis\Desktop\Applied Robotics\Report\Codes\TestML\MLtest_unripe.xlsx")

# === Constants ===
target_size = (224, 224)
true_fruit, true_ripeness = "tomato", "unripe"  # For unripe classification
class_names = [
    'banana_ripeness_ripe', 'banana_ripeness_unripe', 'banana_ripeness_rotten',
    'tomato_ripeness_ripe', 'tomato_ripeness_unripe', 'tomato_ripeness_rotten',
    'strawberry_ripeness_ripe', 'strawberry_ripeness_unripe', 'strawberry_ripeness_rotten'
]

# === Step 1: Resize all images ===
print("🔄 Resizing images...")
for img_path in image_dir.glob("*.*"):
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size)
            img.save(img_path)
        except Exception as e:
            print(f"❌ Failed to resize {img_path.name}: {e}")

# === Step 2: Load model ===
print("📦 Loading model...")
model = load_model(model_path)

# === Step 3: Predict and collect results ===
print("🔍 Running predictions...\n")
results = []
valid_image_paths = [p for p in image_dir.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

for idx, img_path in enumerate(valid_image_paths, start=1):
    try:
        print(f"[{idx}] Processing: {img_path.name}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        label_idx = np.argmax(pred)
        label = class_names[label_idx]
        confidence = float(np.max(pred))

        predicted_fruit, predicted_ripeness = label.split('_ripeness_')
        misclassified = int((predicted_fruit != true_fruit) or (predicted_ripeness != true_ripeness))

        results.append([
            img_path.name, predicted_fruit, predicted_ripeness, f"{confidence:.2f}",
            true_fruit, true_ripeness, misclassified
        ])

    except Exception as e:
        print(f"[{idx}] ❌ Error with {img_path.name}: {e}")

# === Step 4: Save to Excel ===
df = pd.DataFrame(results, columns=[
    "Image", "Predicted_Fruit", "Predicted_Ripeness", "Confidence",
    "True_Fruit", "True_Ripeness", "Misclassified"
])
df["Misclassified"] = df["Misclassified"].astype(int)
df.to_excel(output_excel, index=False)
print(f"\n✅ Excel saved: {output_excel}")

# === Step 5: Visualize misclassified images ===
def show_misclassified_images(df, base_path, num=5):
    print(f"\n📸 Displaying top {num} misclassified images...\n")
    misclassified_df = df[df["Misclassified"] == 1]
    if misclassified_df.empty:
        print("🎯 No misclassifications found.")
        return

    for _, row in misclassified_df.head(num).iterrows():
        img_path = base_path / row['Image']
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Pred: {row['Predicted_Fruit']} ({row['Predicted_Ripeness']}) [{row['Confidence']}]\n"
                      f"True: {row['True_Fruit']} ({row['True_Ripeness']})")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not load image {img_path.name}: {e}")

# === Call the visualizer ===
show_misclassified_images(df, image_dir, num=5)
