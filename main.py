import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from moviepy.editor import ImageSequenceClip

# Loading the FLIM model
model = hub.load("https://tfhub.dev/google/film/1")

# Resize the images to have the same dimensions
def resize_image(image, target_height, target_width):
    return tf.image.resize(image, [target_height, target_width])

# Load and decode image
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_image(img_path: str):
    image_data = tf.io.read_file(img_path)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

# Paths to your images
image_1_url = "D:/PROJECTS/SIH/Backend/cloud1.png"
image_2_url = "D:/PROJECTS/SIH/Backend/cloud2.png"

# Load the images
image1 = load_image(image_1_url)
image2 = load_image(image_2_url)

# Resize images to the same shape
target_height, target_width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
image1 = resize_image(image1, target_height, target_width)
image2 = resize_image(image2, target_height, target_width)

# Generate the mid-frame using the model
time = np.array([0.5], dtype=np.float32)
input = {
    'time': np.expand_dims(time, axis=0), 
    'x0': np.expand_dims(image1, axis=0), 
    'x1': np.expand_dims(image2, axis=0)
}
mid_frame = model(input)

# Convert the output frames to uint8 format
frames = [image1.numpy(), mid_frame['image'][0].numpy(), image2.numpy()]

# media.show_images(frames, titles=['input image one', 'generated image', 'input image two'], height=250)

# def show_images(frames, titles=None, height=250):
#     import matplotlib.pyplot as plt
#     n = len(frames)
#     fig, axes = plt.subplots(1, n, figsize=(15, 5))
#     for i in range(n):
#         axes[i].imshow(frames[i])
#         axes[i].axis('off')
#         if titles:
#             axes[i].set_title(titles[i])
#     plt.show()

# Show the images
# show_images(frames, titles=['Input Image 1', 'Generated Image', 'Input Image 2'])

frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]

# Create a video from the frames
clip = ImageSequenceClip(frames_uint8, fps=3)
clip.write_videofile("output_video.mp4", codec="libx264")
