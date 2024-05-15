import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_remove_periodic_noise(image):
    # Load the image in grayscale
    

    # Get the image dimensions
    rows, cols = image.shape

    # Apply the Fourier Transform
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    threshold = np.max(magnitude_spectrum) * 0.92  # Adjust this threshold as needed
    center_row, center_col = rows // 2, cols // 2

    # Visualize the magnitude spectrum
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121), plt.imshow(image, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Detect peaks in the magnitude spectrum
    peaks = np.where(magnitude_spectrum > threshold, 1, 0)

    # Get the coordinates of the peaks
    peak_coordinates = np.argwhere(peaks == 1)

    if len(list(peak_coordinates)) == 1:
      return False

    # Remove the peaks from the frequency domain except the DC component
    for coord in peak_coordinates:
        if coord[0] != center_row or coord[1] != center_col:
            dft_shift[coord[0], coord[1]] = 0

    # Shift back (inverse FFT shift)
    dft_ishift = np.fft.ifftshift(dft_shift)

    # Perform the inverse Fourier Transform to get the modified image
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the image to display it correctly
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Visualize the result
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    return True



img = cv2.imread('TC/11.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(detect_and_remove_periodic_noise(img))

#
#import os
#from google.colab import drive
#drive.mount('/content/drive')

# Step 2: Define the path to the shortcut folder
#folder_path = '/content/drive/My Drive/CSE483 Sp24 Project Test Cases'

# Step 5: Process each image in the folder
#for filename in os.listdir(folder_path):
 #   if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
  #      image_path = os.path.join(folder_path, filename)
   #     print(f'Processing {image_path}')
       # processed_image = detect_and_remove_periodic_noise(image_path)
        # # Save the processed image if needed
        # output_path = os.path.join(folder_path, 'processed_' + filename)
        # cv2.imwrite(output_path, processed_image)
        # print(f'Saved processed image to {output_path}')