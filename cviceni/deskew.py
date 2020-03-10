import cv2
import numpy as np
import scipy
import scipy.misc


def deskew_fft(image, range_min=-15, range_max=15, blur_size=3, y_width=6, x_width=14):
    ft = np.fft.fft2(image)
    ftshift = np.fft.fftshift(ft)
    spek = 20 * np.log(np.abs(ftshift))
    # Zmenseni velikosti rotovane casti spektra, zrychluje vypocet + je presnejsi
    spect_center = (spek[np.uint32(spek.shape[0]/2)-np.uint32(spek.shape[0]/y_width):np.uint32(spek.shape[0]/2)+np.uint32(spek.shape[0]/y_width),
                         np.uint32(spek.shape[1]/2)-np.uint32(spek.shape[1]/x_width):np.uint32(spek.shape[1]/2)+np.uint32(spek.shape[1]/x_width)])
    maxS = 0
    angle = 0
    for i in range(range_min, range_max):
        imr = skimage.transform.rotate(spect_center, i*0.1)
        temp = np.max(np.sum(imr, 0))
        if temp > maxS:
            maxS = temp
            angle = i*0.1
    return angle


def deskew(image, y_res=(16, 48), x_res=(10, 20), tiles_perct=0.2):
    tiles = []
    border_y = int(image.shape[0]/y_res[0])
    border_x = int(image.shape[1]/x_res[0])
    tile_height = int(image.shape[0]/y_res[1])
    tile_width = int(image.shape[1]/x_res[1])
    for y in range(0, image.shape[0] - border_y, tile_height):
        y2 = y + border_y
        for x in range(0, image.shape[1] - border_x, tile_width):
            x2 = x + border_x
            tiles.append((y, y2, x, x2, np.mean(image[y:y2, x:x2])))
    tiles.sort(key=itemgetter(4))
    n = int(len(tiles)*tiles_perct)
    angle_mean = 0
    angle = 0
    for i in range(n):
        y = tiles[i][0]
        y2 = tiles[i][1]
        x = tiles[i][2]
        x2 = tiles[i][3]
        part = image[y:y2, x:x2]
        angle = deskew_fft(part)
        angle_mean = angle_mean + angle
    return angle_mean / float(n)


def apply_deskew(image):
    angle = 0.0
    if len(image.shape) == 2:
        angle = deskew(image)
        image_deskew = rotate(image, angle)
    elif len(image.shape) == 3:
        angle_r = deskew(image[:, :, 0])
        angle_g = deskew(image[:, :, 1])
        angle_b = deskew(image[:, :, 2])
        angle = (angle_r + angle_g + angle_b)/3.0
        image_deskew_r = rotate(image[:, :, 0], angle)
        image_deskew_g = rotate(image[:, :, 1], angle)
        image_deskew_b = rotate(image[:, :, 2], angle)
        image_deskew = np.dstack([image_deskew_r, image_deskew_g, image_deskew_b])
    else:
        print("Vstup musi byt RGB nebo sedotonovy obraz")
    return image_deskew, angle


# img = cv2.imread("path", 0)
# img_np = np.array(img)
# img_np, angle = apply_deskew(img_np)
