import csv
import os
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Initialize Flask app
app = Flask(__name__)

# Set the folder for uploaded files (inside the static folder)
UPLOAD_FOLDER = 'static/uploads'
ADV_FOLDER = 'static/adversarial'  # ✅ Folder to save adversarial images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ADV_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ADV_FOLDER'] = ADV_FOLDER

# Load pre-trained models
generator = load_model('Models/gan.h5')
discriminator = load_model('Models/discriminator.h5')
dae = tf.keras.models.load_model('Models/denoising_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU enabled:", physical_devices[0])
else:
    print("⚠️ No GPU found, using CPU")

# Define image size for resizing
image_size = (256, 256)
batch_size = 32

# Helper function to preprocess image
def preprocess_image(img):
    img = img.resize(image_size)
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ✅ Helper function to save adversarial image
def save_adversarial_image(adv_tensor, filename):
    base_name = os.path.splitext(secure_filename(filename))[0]
    save_name = f"adv_{base_name}.png"
    save_path = os.path.join(ADV_FOLDER, save_name)

    # Convert tensor → numpy → uint8 PIL image
    adv_array = adv_tensor.numpy()
    adv_array = (adv_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(adv_array).save(save_path)

    return save_name

# Adversarial example generation function (IFGSM)
def generate_adversarial_examples(dae, batch_images, epsilon=0.1, num_iterations=10):
    batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
    adv_examples = tf.identity(batch_images)
    adv_examples = tf.Variable(adv_examples, dtype=tf.float32)

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_examples)
            reconstructions = dae(adv_examples)
            reconstruction_loss = tf.reduce_mean((adv_examples - reconstructions) ** 2)

        gradients = tape.gradient(reconstruction_loss, adv_examples)
        signed_gradients = tf.sign(gradients)
        adv_examples.assign_add(epsilon * signed_gradients)
        adv_examples.assign(tf.clip_by_value(adv_examples, 0.0, 1.0))

    return adv_examples

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No files part"})

    files = request.files.getlist('images')

    images = []
    saved_image_paths = []
    filenames = []

    for file in files:
        filename = secure_filename(file.filename)
        filenames.append(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        saved_image_paths.append(file_path)

        # Reopen saved file for preprocessing (stream already consumed by save)
        img = Image.open(file_path).convert('RGB')
        img_data = preprocess_image(img)
        images.append(img_data)

    # Convert list to batch
    batch_images = np.vstack(images)

    # Generate adversarial examples
    adv_examples = generate_adversarial_examples(dae, batch_images)

    # ✅ Save each adversarial image
    adv_image_names = []
    for i, filename in enumerate(filenames):
        adv_name = save_adversarial_image(adv_examples[i], filename)
        adv_image_names.append(adv_name)
        print(f"✅ Saved adversarial image: {adv_name}")

    # Get predictions from discriminator
    real_scores = discriminator.predict(batch_images)
    adv_scores = discriminator.predict(adv_examples)

    # Compute reconstruction errors
    reconstructions = dae.predict(batch_images, batch_size=batch_size)
    reconstruction_errors = np.mean((batch_images - reconstructions) ** 2, axis=(1, 2, 3))

    print("Real Scores:", real_scores)
    print("Adversarial Scores:", adv_scores)
    print("Reconstruction Errors:", reconstruction_errors)

    # Thresholds
    threshold_reconstruction_error = 0.05
    threshold_real_score = 0.5

    # Flag poisoned images
    poisoned_indices = np.where(
        (real_scores.flatten() < threshold_real_score) &
        (reconstruction_errors > threshold_reconstruction_error)
    )[0]

    any_poisoned = len(poisoned_indices) > 0

    # Prepare results
    results = []
    for i, file in enumerate(files):
        is_poisoned = i in poisoned_indices
        results.append({
            'Image_Name': filenames[i],
            'Image_Url': f"uploads/{filenames[i]}",                    # ✅ original image URL
            'Adv_Image_Url': f"adversarial/{adv_image_names[i]}",      # ✅ adversarial image URL
            'Adv_Image_Name': adv_image_names[i],
            'Is_Poisoned': is_poisoned,
            'Real_Score': round(float(real_scores[i][0]), 4),
            'Adversarial_Score': round(float(adv_scores[i][0]), 4),
            'Reconstruction_Error': round(float(reconstruction_errors[i]), 6)
        })

    return render_template('results.html', results=results, any_poisoned=any_poisoned)

if __name__ == '__main__':
    app.run(debug=True)