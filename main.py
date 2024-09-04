import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import imageio
from PIL import Image
import os
import warnings
import logging
from typing import Generator, Iterable, List
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_image(image_path: str):
    image_data = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

def resize_image(image, target_height, target_width):
    return tf.image.resize(image, [target_height, target_width])

def save_frames_as_video(frames, video_path, fps=3):
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)  # Convert frame to uint8
            writer.append_data(frame_uint8)

def _pad_to_align(x, align):
    assert np.ndim(x) == 4
    assert align > 0, 'align must be a positive number.'

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
        self._model = hub.load("https://tfhub.dev/google/film/1")
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
        return image.numpy()

def _recursive_generator(frame1: np.ndarray, frame2: np.ndarray, num_recursions: int, interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
    if num_recursions == 0:
        yield frame1
    else:
        time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator)

def interpolate_recursively(frames: List[np.ndarray], num_recursions: int, interpolator: Interpolator) -> Iterable[np.ndarray]:
    n = len(frames)
    for i in range(1, n):
        yield from _recursive_generator(frames[i - 1], frames[i], num_recursions, interpolator)
    yield frames[-1]

# Paths to the images
image_1_url = "D:/PROJECTS/SIH/Backend/cloud1.jpg"
image_2_url = "D:/PROJECTS/SIH/Backend/cloud2.jpg"

# Time array
time = np.array([0.5], dtype=np.float32)

# Load images
image1 = load_image(image_1_url)
image2 = load_image(image_2_url)

# Ensure both images have the same dimensions
target_height, target_width = image1.shape[:2]
image2 = resize_image(image2, target_height, target_width)

# Create an interpolator instance
times_to_interpolate = 6
interpolator = Interpolator()

# Generate interpolated frames
input_frames = [image1, image2]
frames = list(interpolate_recursively(input_frames, times_to_interpolate, interpolator))

# Save the frames as a video in mp4 format
save_frames_as_video(frames, "interpolated_video.mp4", fps=30)

print(f'video with {len(frames)} frames')
