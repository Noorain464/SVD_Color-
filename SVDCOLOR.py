from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
filepath = input()
color_image = imread(filepath)
plt.imshow(color_image)
color_image.shape
np.shape(color_image)
color_image
color_image_red = color_image[:,:,0]
color_image_blue = color_image[:,:,1]
color_image_green = color_image[:,:,2]
color_image_red
color_image_red
color_image_blue
color_image_green
U_red, s_red, V_red = np.linalg.svd(color_image_red)
U_red, s_red, V_red
U_blue, s_blue, V_blue = np.linalg.svd(color_image_blue)
U_blue, s_blue, V_blue
U_green, s_green, V_green = np.linalg.svd(color_image_green)
U_green, s_green, V_green
np.diag(s_red)
rank=5 # here the rank is the number of singular values we use to compress our image
compress_red=U_red[:, :rank] @ np.diag(s_red[:rank])@V_red[:rank, :]  #use @ instead of * for matrix mult in Python!
compress_red
compress_red=compress_red.astype(int)
compress_red
def compress(rank):
    compress_red = U_red[:, :rank] @ np.diag(s_red[:rank]) @ V_red[:rank, :]
    compress_red = np.clip(compress_red, 0, 255).astype(np.uint8)
    
    compress_blue = U_blue[:, :rank] @ np.diag(s_blue[:rank]) @ V_blue[:rank, :]
    compress_blue = np.clip(compress_blue, 0, 255).astype(np.uint8)

    compress_green = U_green[:, :rank] @ np.diag(s_green[:rank]) @ V_green[:rank, :]
    compress_green = np.clip(compress_green, 0, 255).astype(np.uint8)

    compressed_array = np.stack((compress_red, compress_blue, compress_green), axis=2)


    compressed_array=np.stack((compress_red, compress_blue, compress_green), axis=2)
    plt.imshow(compressed_array)
    plt.show()
compress(20)

