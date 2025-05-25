import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# Nastavení
CONTENT_PATH = "content.jpg"
STYLE_PATH = "style.jpg"
OUTPUT_PATH = "result.png"
IMAGE_SIZE = (384, 512)
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2

# Funkce pro načtení a zpracování obrázků
def load_and_process_image(path):
    # Načte obrázek, převede na RGB, změní velikost a připraví pro VGG19
    img = Image.open(path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img[tf.newaxis, :])

def deprocess_image(img):
    # Převod zpět z VGG19 formátu do běžného obrázku
    img = img.numpy().squeeze()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img / 255.0, 0, 1)

# Výběr vrstev z VGG19 pro extrakci rysů
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
num_content = len(content_layers)
num_style = len(style_layers)

#  Vytvoření modelu založeného na VGG19
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Výpočet Gramovy matice
def gram_matrix(tensor):
    x = tf.squeeze(tensor)
    x = tf.reshape(x, (-1, x.shape[-1]))
    return tf.matmul(x, x, transpose_a=True)

# Výpočet ztrátové funkce
def compute_loss(model, outputs, style_targets, content_targets):
    style_outputs = outputs[:num_style]
    content_outputs = outputs[num_style:]

    # Stylová ztráta
    style_loss = tf.add_n([
        tf.reduce_mean((gram_matrix(style_outputs[i]) - gram_matrix(style_targets[i])) ** 2)
        for i in range(num_style)
    ])
    style_loss *= STYLE_WEIGHT / num_style

    # Obsahová ztráta
    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[i] - content_targets[i]) ** 2)
        for i in range(num_content)
    ])
    content_loss *= CONTENT_WEIGHT / num_content

    total_loss = style_loss + content_loss  # kombinovaná ztráta
    return total_loss

# Jeden krok trénovací smyčky (gradient descent nad vstupním obrázkem)
@tf.function()
def train_step(image, model, style_targets, content_targets, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(image)
        loss = compute_loss(model, outputs, style_targets, content_targets)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, -103.939, 255.0 - 103.939))

# Hlavní funkce pro spuštění převodu stylu
def run_style_transfer(content_path, style_path, output_path, epochs=50):
    # Načtení a zpracování obrázků
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    model = get_model()

    # Extrakce stylových a obsahových cílů
    style_targets = model(style_image)[:num_style]
    content_targets = model(content_image)[num_style:]

    # Počáteční obrázek: kopie obsahového obrázku
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    start = time.time()
    for i in range(epochs):
        train_step(generated_image, model, style_targets, content_targets, optimizer)
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{epochs}")

    print(f"Finished in {time.time() - start:.1f}s")
    final_img = deprocess_image(generated_image)
    plt.imsave(output_path, final_img)
    print(f"Saved result to {output_path}")

# Spuštění
if __name__ == "__main__":
    run_style_transfer(CONTENT_PATH, STYLE_PATH, OUTPUT_PATH)