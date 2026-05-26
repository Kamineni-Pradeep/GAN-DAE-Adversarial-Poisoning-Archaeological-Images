import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
from tensorflow.keras.losses import MeanSquaredError

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained DAE and GAN models
generator = load_model('Models/gan.h5')
discriminator = load_model('Models/discriminator.h5')
dae = tf.keras.models.load_model('Models/denoising_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})

# Enable GPU memory growth (prevents TensorFlow from taking all GPU memory)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU enabled:", physical_devices[0])
else:
    print("No GPU found, using CPU")

# Define image size for resizing
image_size = (256, 256)
batch_size = 32

# Folder to save adversarial images
ADV_SAVE_FOLDER = 'static/adversarial_images'
os.makedirs(ADV_SAVE_FOLDER, exist_ok=True)

# Helper function to preprocess image
def preprocess_image(img):
    img = img.resize(image_size)
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Helper function to save adversarial image to disk
def save_adversarial_image(adv_array, filename):
    # Convert from float [0,1] back to uint8 [0,255]
    adv_img = (adv_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(adv_img)
    
    # Build save path
    base_name = os.path.splitext(filename)[0]
    save_name = f"adv_{base_name}.png"
    save_path = os.path.join(ADV_SAVE_FOLDER, save_name)
    
    pil_img.save(save_path)
    return save_name  # Return just the filename for use in HTML template

# Define adversarial example generation function (IFGSM)
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
    return render_template('index1.html')

# Endpoint to accept multiple images and display results in HTML
@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No files part"})

    files = request.files.getlist('images')

    # Process each image
    images = []
    filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        filenames.append(filename)
        img = Image.open(file.stream).convert('RGB')  # Ensure RGB
        img = preprocess_image(img)
        images.append(img)

    # Convert list to a batch
    batch_images = np.vstack(images)

    # Generate adversarial examples
    adv_examples = generate_adversarial_examples(dae, batch_images)

    # ✅ SAVE each adversarial image to disk
    adv_image_names = []
    for i, filename in enumerate(filenames):
        adv_array = adv_examples[i].numpy()  # Convert tensor to numpy
        adv_name = save_adversarial_image(adv_array, filename)
        adv_image_names.append(adv_name)

    # Get predictions from the discriminator
    real_scores = discriminator.predict(batch_images)
    adv_scores = discriminator.predict(adv_examples)

    # Compute reconstruction errors for the DAE
    reconstructions = dae.predict(batch_images, batch_size=batch_size)
    reconstruction_errors = np.mean((batch_images - reconstructions) ** 2, axis=(1, 2, 3))

    # Set thresholds for flagging potential poisoning
    threshold_reconstruction_error = 0.05
    threshold_real_score = 0.5

    # Flag poisoned images based on both reconstruction error and discriminator score
    poisoned_indices = np.where(
        (real_scores.flatten() < threshold_real_score) &
        (reconstruction_errors > threshold_reconstruction_error)
    )[0]

    # Prepare results to pass to the HTML template
    results = []
    for i, filename in enumerate(filenames):
        is_poisoned = i in poisoned_indices
        results.append({
            'image_name': filename,
            'adv_image_name': adv_image_names[i],      # ✅ Path to saved adversarial image
            'adv_image_url': f"adversarial_images/{adv_image_names[i]}",  # ✅ URL for HTML display
            'is_poisoned': is_poisoned,
            'real_score': round(float(real_scores[i][0]), 4),
            'adv_score': round(float(adv_scores[i][0]), 4),
            'reconstruction_error': round(float(reconstruction_errors[i]), 6)
        })

    return render_template('results1.html', results=results)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)