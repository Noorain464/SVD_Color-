from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def embed_watermark(image, watermark_text):
    watermark = np.array([[ord(c) for c in watermark_text]])
    watermark_height, watermark_width = watermark.shape
    image_height, image_width = image.shape[:2]
    scale_factor_height = image_height // watermark_height
    scale_factor_width = image_width // watermark_width
    watermark_resized = np.tile(watermark, (scale_factor_height, scale_factor_width))
    alpha = 0.1  # Adjust alpha value to control watermark visibility
    watermarked_image = image.copy()
    watermarked_image[:, :, 0] = (1 - alpha) * watermarked_image[:, :, 0] + alpha * watermark_resized
    return watermarked_image


def compress_image(filepath, rank):
    color_image = imread(filepath)
    color_image_red = color_image[:,:,0]
    color_image_blue = color_image[:,:,1]
    color_image_green = color_image[:,:,2]

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
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.show()

def main():
    filepath = input("Enter the path to the image file: ")
    rank = int(input("Enter the rank for compression (e.g., 5, 10, 20): "))
    watermark_text = input("Enter the watermark text: ")
    
    compressed_image = compress_image(filepath, rank)
    watermarked_image = embed_watermark(compressed_image, watermark_text)
    
    show_compressed_image(watermarked_image)

if __name__ == "__main__":
    main()