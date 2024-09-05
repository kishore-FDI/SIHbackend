from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from moviepy.editor import ImageSequenceClip
import os

app = Flask(__name__)

# Load the FLIM model once when the app starts
model = hub.load("https://tfhub.dev/google/film/1")

# Function to resize the images
def resize_image(image, target_height, target_width):
    return tf.image.resize(image, [target_height, target_width])

# Function to load and decode images
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_image(img_path: str):
    image_data = tf.io.read_file(img_path)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

# Wrapper class and functions for frame interpolation
def _pad_to_align(x, align):
    height, width = x.shape[-3:-1]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    bbox_to_pad = {
        'offset_height': height_to_pad // 2,
        'offset_width': width_to_pad // 2,
        'target_height': height + height_to_pad,
        'target_width': width + width_to_pad
    }
    padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
    bbox_to_crop = {
        'offset_height': height_to_pad // 2,
        'offset_width': width_to_pad // 2,
        'target_height': height,
        'target_width': width
    }
    return padded_x, bbox_to_crop

class Interpolator:
    def __init__(self, align: int = 64) -> None:
        self._model = model
        self._align = align

    def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align(x0, self._align)
            x1, _ = _pad_to_align(x1, self._align)

        inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        image = result['image']

        if self._align is not None:
            image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image

def _recursive_generator(frame1: np.ndarray, frame2: np.ndarray, num_recursions: int, interpolator: Interpolator):
    if num_recursions == 0:
        yield frame1
    else:
        time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator)

def interpolate_recursively(frames, num_recursions: int, interpolator: Interpolator):
    n = len(frames)
    for i in range(1, n):
        yield from _recursive_generator(frames[i - 1], frames[i], num_recursions, interpolator)
    yield frames[-1]

@app.route('/generate_video', methods=['POST'])
def generate_video():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please upload both images"}), 400
    
    image1_file = request.files['image1']
    image2_file = request.files['image2']
    
    # Save uploaded images to temporary files
    image1_path = 'image1.png'
    image2_path = 'image2.png'
    image1_file.save(image1_path)
    image2_file.save(image2_path)
    
    # Load the images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Resize images to the same shape
    target_height, target_width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
    image1 = resize_image(image1, target_height, target_width)
    image2 = resize_image(image2, target_height, target_width)

    # Interpolation
    interpolator = Interpolator()
    input_frames = [image1, image2]
    times_to_interpolate = 1
    frames = list(interpolate_recursively(input_frames, times_to_interpolate, interpolator))

    # Convert Tensor frames to NumPy arrays and then to uint8 format for saving
    frames_uint8 = [(frame.numpy() * 255).astype(np.uint8) for frame in frames]

    # Save the video
    output_video_path = 'output_video.mp4'
    clip = ImageSequenceClip(frames_uint8, fps=60)
    clip.write_videofile(output_video_path, codec="libx264")

    # Clean up the images
    os.remove(image1_path)
    os.remove(image2_path)

    # Send the video file as a response
    return send_file(output_video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
