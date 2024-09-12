import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import os
from datetime import datetime

def embed_fingerprint(image, scale=0.001):
    
    fingerprint = generate_fingerprint(image.shape[:2], scale)
    fingerprint_with_channels = np.expand_dims(fingerprint) 
    watermarked_image = image + fingerprint_with_channels
    return watermarked_image

def generate_fingerprint(image_shape, scale):
    fingerprint_height, fingerprint_width = image_shape
    fingerprint = np.random.rand(fingerprint_height, fingerprint_width)
    fingerprint *= scale
    return fingerprint

def compress_image(filepath, quality_retention):
    
    color_image = imread(filepath)
    color_image_red = color_image[:, :, 0]
    color_image_blue = color_image[:, :, 1]
    color_image_green = color_image[:, :, 2]

    U_red, s_red, V_red = np.linalg.svd(color_image_red)
    U_blue, s_blue, V_blue = np.linalg.svd(color_image_blue)
    U_green, s_green, V_green = np.linalg.svd(color_image_green)

    cumulative_sum = np.cumsum(s_red)
    cumulative_percentage = cumulative_sum / np.sum(s_red)


    rank = np.argmax(cumulative_percentage >= quality_retention)

    compress_red = U_red[:, :rank] @ np.diag(s_red[:rank]) @ V_red[:rank, :]
    compress_blue = U_blue[:, :rank] @ np.diag(s_blue[:rank]) @ V_blue[:rank, :]
    compress_green = U_green[:, :rank] @ np.diag(s_green[:rank]) @ V_green[:rank, :]

    compress_red = np.clip(compress_red, 0, 255).astype(np.uint8)
    compress_blue = np.clip(compress_blue, 0, 255).astype(np.uint8)
    compress_green = np.clip(compress_green, 0, 255).astype(np.uint8)

    compressed_array = np.stack((compress_red, compress_blue, compress_green), axis=2)
    return compressed_array, cumulative_percentage[rank]

def show_compressed_image(compressed_image):
    
    plt.imshow(compressed_image)
    plt.show()

def plot_cumulative_percentage(cumulative_percentage):
    
    plt.plot(cumulative_percentage, marker='o', linestyle='-')
    plt.title('Cumulative Percentage of Information Retained')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Percentage')
    plt.show()

def verify_fingerprint(image, scale):
    fingerprint = generate_fingerprint(image.shape[:2], scale)
    fingerprint_with_channels = np.expand_dims(fingerprint, axis=-1)

    watermarked_image = image + fingerprint_with_channels
    watermarked_image_clipped = np.clip(watermarked_image, 0, 1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    watermarked_image_path = f"watermarked_image_{timestamp}.png"
    plt.imsave(watermarked_image_path, watermarked_image_clipped)
    
    loaded_watermarked_image = imread(watermarked_image_path)
    
    if np.array_equal(watermarked_image_clipped, loaded_watermarked_image):
        print("Fingerprint not detected.")
    else:
        print("Fingerprint detected.")



def main():
    filepath = input("Enter image file path: ")
    quality_retention = float(input("Enter the desired quality retention percentage (0 to 100): "))
    quality_retention /= 100 
    scale = float(input("Enter the scale factor for fingerprinting (e.g., 0.001): "))

    compressed_image, retained_percentage = compress_image(filepath, quality_retention)
    show_compressed_image(compressed_image)
    plot_cumulative_percentage(retained_percentage)

    verify = input("Do you want to verify the presence of the fingerprint? (yes/no): ")
    if verify == "yes":
        verify_fingerprint(compressed_image, scale)

if __name__ == "__main__":
    main()

