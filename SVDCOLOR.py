import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

def embed_fingerprint(image, scale=0.001):
    """
    Embeds a fingerprint into the image.
    """
    fingerprint = generate_fingerprint(image.shape[:2], scale)
    fingerprint_with_channels = np.expand_dims(fingerprint, axis=-1)  # Add a channel axis
    watermarked_image = image + fingerprint_with_channels
    return watermarked_image

def generate_fingerprint(image_shape, scale):
    fingerprint_height, fingerprint_width = image_shape
    fingerprint = np.random.rand(fingerprint_height, fingerprint_width)
    fingerprint *= scale
    return fingerprint

def compress_image(filepath, rank):
    """
    Compresses the image using Singular Value Decomposition (SVD).
    """
    color_image = imread(filepath)
    color_image_red = color_image[:, :, 0]
    color_image_blue = color_image[:, :, 1]
    color_image_green = color_image[:, :, 2]

    U_red, s_red, V_red = np.linalg.svd(color_image_red)
    U_blue, s_blue, V_blue = np.linalg.svd(color_image_blue)
    U_green, s_green, V_green = np.linalg.svd(color_image_green)

    compress_red = U_red[:, :rank] @ np.diag(s_red[:rank]) @ V_red[:rank, :]
    compress_blue = U_blue[:, :rank] @ np.diag(s_blue[:rank]) @ V_blue[:rank, :]
    compress_green = U_green[:, :rank] @ np.diag(s_green[:rank]) @ V_green[:rank, :]

    compress_red = np.clip(compress_red, 0, 255).astype(np.uint8)
    compress_blue = np.clip(compress_blue, 0, 255).astype(np.uint8)
    compress_green = np.clip(compress_green, 0, 255).astype(np.uint8)

    compressed_array = np.stack((compress_red, compress_blue, compress_green), axis=2)
    return compressed_array

def show_compressed_image(compressed_image):
    """
    Displays the compressed image.
    """
    plt.imshow(compressed_image)
    plt.title('Compressed Image')
    plt.show()

def main():
    filepath = input("Enter the path to the image file: ")
    rank = int(input("Enter the rank for compression (e.g., 5, 10, 20): "))
    scale = float(input("Enter the scale factor for fingerprinting (e.g., 0.001): "))

    compressed_image = compress_image(filepath, rank)
    fingerprinted_image = embed_fingerprint(compressed_image, scale)

    plt.imshow(fingerprinted_image)
    plt.title('Fingerprinted Image')
    plt.show()

if __name__ == "__main__":
    main()
