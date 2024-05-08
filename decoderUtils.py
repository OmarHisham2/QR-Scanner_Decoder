import cv2
import numpy as np
import matplotlib.pyplot as plt
import reedsolo as rs
import utils as ut
from enum import Enum

class Encoding(Enum):
    Numeric = "utf-8"
    ALPHA = "ascii"
    BYTE = "iso-8859-1"
    KANJI = "utf-8"             #todo

EncodingTable={
    "0001":Encoding.Numeric,
    "0010":Encoding.ALPHA,        # upper case only and some special characters including space
    "0100":Encoding.BYTE,   # byte
    "1000":Encoding.KANJI         #13 byte
}


MASKS = {
    "000": lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
    "001": lambda i, j: (i / 2 + j / 3) % 2 == 0,
    "010": lambda i, j: ((i * j) % 3 + i + j) % 2 == 0,
    "011": lambda i, j: ((i * j) % 3 + i * j) % 2 == 0,
    "100": lambda i, j: i % 2 == 0,
    "101": lambda i, j: (i + j) % 2 == 0,
    "110": lambda i, j: (i + j) % 3 == 0,
    "111": lambda i, j: j % 3 == 0,
}

UP8, UP4, DOWN8, DOWN4, CW8, CCW8 = range(6)
block_starting_indices = [
    [21-7,  21-1,  UP8],
    [21-11, 21-1,  CCW8],
    [21-10, 21-3,  DOWN8],
    [21-6,  21-3,  DOWN8],
    [21-2,  21-3,  CW8],
    [21-3,  21-5,  UP8],
    [21-7,  21-5,  UP8],
    [21-11, 21-5,  CCW8],
    [21-10, 21-7,  DOWN8],
    [21-6,  21-7,  DOWN8],
    [21-2,  21-7,  CW8],
    [21-3,  21-9,  UP8],
    [21-7,  21-9,  UP8],
    [21-11, 21-9,  UP8],
    [21-16, 21-9,  UP8],
    [21-20, 21-9,  CCW8],
    [21-19, 21-11, DOWN8],
    [21-14, 21-11, DOWN4],  # Special 4-byte block, reserved for END (if exists!)
    [21-12, 21-11, DOWN8],
    [21-8,  21-11, DOWN8],
    [21-4,  21-11, DOWN8],
    [21-9,  21-13, UP8],
    [21-12, 21-16, DOWN8],
    [21-9,  21-18, UP8],
    [21-12, 21-20, DOWN8],
]


def removeQuietZone(image,showImage=False):
    """ 
    Remove the quiet zone from the QR code image.
    * image: thresholded qr code image
    """
    start_row = -1
    start_col = -1
    end_row = -1
    end_col = -1

    for row_index, row in enumerate(image):
        for pixel in row:
            if pixel != 255:
                start_row = row_index
                break
        if start_row != -1:
            break

    for row_index, row in enumerate(image[::-1]):
        for pixel in row:
            if pixel != 255:
                end_row = image.shape[0] - row_index
                break
        if end_row != -1:
            break

    for col_index, col in enumerate(cv2.transpose(image)):
        for pixel in col:
            if pixel != 255:
                start_col = col_index
                break
        if start_col != -1:
            break

    for col_index, col in enumerate(cv2.transpose(image)[::-1]):
        for pixel in col:
            if pixel != 255:
                end_col = image.shape[1] - col_index
                break
        if end_col != -1:
            break
    
    if showImage:
        fig = plt.figure(figsize=(3, 3));
        plt.xticks([], []);
        plt.yticks([], []);
        fig.get_axes()[0].spines[:].set_color('red');
        fig.get_axes()[0].spines[:].set_linewidth(40);
        fig.get_axes()[0].spines[:].set_position(("outward", 20))
        plt.title('QR code without quiet zone', y = 1.15, color='red');
        plt.imshow(image[start_row:end_row, start_col:end_col], cmap='gray'); 
        plt.show()

    return image[start_row:end_row, start_col:end_col]


def gridQrCode(image,ratio=0.5,showImage=False):
    """
    Divide the QR code image into 21x21 grid.
    * image: removed quiet zone qr code thresholded image
    """
    # Todo: add feature to change ratio
        
    # grid_cells_num = 21

    #check if qr code dimensions without QZ is square , if not make it square
    # newsize = max(image.shape[0], image.shape[1])
    # #ensure that the new size is multiple of 21:
    # newsize = grid_cells_num * np.ceil(newsize / grid_cells_num).astype(int)
    # image = cv2.resize(image, (newsize, newsize))
    
    image = cv2.resize(image, (21, 21), interpolation=cv2.INTER_AREA)#//128
    
    if showImage:
        _, axes = plt.subplots(21, 21, figsize=(5, 5))
        for i, row in enumerate(axes):
            for j, col in enumerate(row):
                col.imshow([[image[i][j]]], cmap="gray", vmin=0, vmax=1)
                col.get_xaxis().set_visible(False)
                col.get_yaxis().set_visible(False)
                col.spines[:].set_color('red')
        plt.show()

    return image


def invertColors(image):
    """
    Invert the colors of the image.
    * image: thresholded qr code image
    """
    return 1 - image


def getECL(image):
    """
    Get the error correction level from the QR code image.
    * image: qr code image
    """
    return image[8, 0:2]


def getMaskValue(image):
    """
    Get the mask pattern from the QR code image.
    * image: qr code image
    """
    mask = image[8, 2:5]
    mask_str = ''.join([str(c) for c in mask])
    return mask_str


def applyMask(data_start_i, data_start_j, data, mask, direction):
    result = []
    row_offsets = []
    col_offsets = []
    mask_str = ''.join([str(c) for c in mask])
    if (direction in [UP8, UP4]):
        row_offsets = [0,  0, -1, -1, -2, -2, -3, -3]
        col_offsets = [0, -1,  0, -1,  0, -1,  0, -1]
    if (direction in [DOWN8, DOWN4]):
        row_offsets = [0,  0,  1,  1,  2,  2,  3,  3]
        col_offsets = [0, -1,  0, -1,  0, -1,  0, -1]
    if (direction == CW8):
        row_offsets = [0,  0,  1,  1,  1,  1,  0,  0]
        col_offsets = [0, -1,  0, -1, -2, -3, -2, -3]
    if (direction == CCW8):
        row_offsets = [0,  0, -1, -1, -1, -1,  0,  0]
        col_offsets = [0, -1,  0, -1, -2, -3, -2, -3]
    for i, j in zip(row_offsets, col_offsets):
        cell_bit = bool(data[data_start_i+i, data_start_j+j])
        mask_bit = MASKS[mask_str](data_start_i+i, data_start_j+j)
        # Modules corresponding to the dark areas of the mask are inverted.
        result.append(int(not cell_bit if mask_bit else cell_bit))

    return result[:4] if direction in [UP4, DOWN4] else result


def getENCBits(image,mask,messageBits=None):
    enc= applyMask(21-1, 21-1, image, mask, UP4)
    if messageBits:
        messageBits.extend(enc)
    return enc


def getLenght(image, mask,encoding: Encoding):
    data_start_i=23-3
    data_start_j=21-1
    len_bits = applyMask(21-3, 21-1, image, mask, UP8)

    if encoding is Encoding.Numeric:  # len 10
        cell_bit = bool(image[21-7, 21-1])
        mask_bit = MASKS[mask](21-7, 21-1)
        # Modules corresponding to the dark areas of the mask are inverted.
        len_bits.append(int(not cell_bit if mask_bit else cell_bit))
        
        cell_bit = bool(image[21-7, 21-2])
        mask_bit = MASKS[mask](21-7, 21-2)
        # Modules corresponding to the dark areas of the mask are inverted.
        len_bits.append(int(not cell_bit if mask_bit else cell_bit))
    
    if encoding is Encoding.ALPHA: # len 9 21-7 row
        cell_bit = bool(image[21-7, 21-1])
        mask_bit = MASKS[mask](21-7, 21-1)
        # Modules corresponding to the dark areas of the mask are inverted.
        len_bits.append(int(not cell_bit if mask_bit else cell_bit))
    
    # lenght field is 8 bits in both byte and kanji case
    
    len_int = int(''.join([str(bit) for bit in len_bits]), 2)
    return len_int


def getLenghtBits(image, mask):
    len_bits = applyMask(21-3, 21-1, image, mask, UP8)
    return len_bits


def decodeQrFromIndex(image,byte_index,lenght,encoding:Encoding,messageBits,mask):
    for _ in range(lenght):
        start_i, start_j, dir = block_starting_indices[byte_index]
        bits = applyMask(start_i, start_j, image, mask, dir)
        messageBits.extend(bits)
        bit_string = ''.join([str(bit) for bit in bits])
        alpha_char = chr(int(bit_string, 2))
        print(f'{bit_string} (={int(bit_string, 2):03d}) = {alpha_char}')
        byte_index += 1

    # After finishing all the characters, the next 4 bits are expected to be '0000'
    start_i, start_j, dir = block_starting_indices[byte_index]
    bits = applyMask(start_i, start_j, image, mask, dir)
    messageBits.extend(bits)
    bit_string = ''.join([str(bit) for bit in bits])
    print(f'{bit_string} (=END) -- the NULL TERMINATOR, followed by padding and/or ECC')
    byte_index += 1
    # Let's see what the bytes that follow look like
    # There supposedly remain 25-len-1 bytes to be read
    for _ in range(25 - lenght - 1):
        start_i, start_j, dir = block_starting_indices[byte_index]
        bits = applyMask(start_i, start_j, image, mask, dir)
        messageBits.extend(bits)
        bit_string = ''.join([str(bit) for bit in bits])
        alpha_char = chr(int(bit_string, 2))
        print(f'{bit_string} (={int(bit_string, 2):03d}) = {alpha_char}')
        byte_index += 1
        
    # error correction (reedsolomon)
    # For every 8 bits in the extracted messageBits, convert to a byte
    message_bytes = [int("".join(map(str, messageBits[i:i+8])), 2) for i in range(0, len(messageBits), 8)]

    # Create the Reed-Solomon Codec for 7 ECC symbols (again, this is L)
    rsc = rs.RSCodec(nsym=7)
    print(message_bytes)

    # Decode the bytes with the 7-ECC RS Codec
    message_decoded = rsc.decode(message_bytes)
    rsc.maxerrata(verbose=True)

    # In order to extract the actual data, need to convert back to bits
    # Then take as many bytes as indicated by the messageBits length indicator
    # That is AFTER removing the first 12 bytes (of enc and len)
    data_bits = bin(int.from_bytes(message_decoded[0], byteorder='big'))[13:13+lenght*8]

    # Now convert back to bytes and print it lol
    data_bytes = int(data_bits, 2).to_bytes((len(data_bits)+7)//8, 'big')
    print(f'Data in messageBits = "{data_bytes.decode(encoding=encoding)}"')
    print(f'Data should be... "01-Good job!"')
    return (image,data_bytes.decode(encoding="iso-8859-1"))


def decodeQrCode(image,showImage=False):
    """
    Decode the QR code image.
    * image: qr code image
    """
    image = removeQuietZone(image,showImage)
    image = gridQrCode(image,showImage=showImage)
    image = invertColors(image)
    print(image)
    mask = getMaskValue(image)
    messageBits = []
    lenght = getLenght(image, mask,encoding)
    enc_bits = getENCBits(image, mask)
    encoding=EncodingTable[enc_bits]  #keys are strings
    return decodeQrFromIndex(image,0,lenght,encoding,messageBits,mask)
    