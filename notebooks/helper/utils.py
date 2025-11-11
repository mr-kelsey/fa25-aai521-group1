import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure

from random import shuffle, randint


def add_damage(img_path):
    # load image from image path, add damage, return damaged image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    """Adapted from https://stackoverflow.com/a/71904002"""
    random_number_generator = np.random.default_rng()

    # create random noise image
    noise = random_number_generator.integers(0, 255, (height, width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])

    # add mask to input
    return cv2.add(img, mask)

def add_noise(img_path):
    # load image from image path, add noise, return noisy image
    """Adapted from https://github.com/ABAHA15/CleanVision_using_DIV2K/blob/main/Image_denoising_%26_Inpainting.ipynb"""
    img = cv2.imread(img_path)

    def add_gaussian_noise(img, mean=0, sigma=75):
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
        return noisy

    def add_salt_pepper_noise(img, prob=0.08):
        noisy = np.copy(img)
        h, w = img.shape[:2]
        num_salt = int(prob * h * w)
        num_pepper = int(prob * h * w)

        # Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 0

        return noisy

    def add_speckle_noise(img):
        gauss = np.random.randn(*img.shape) * 0.5
        noisy = img + img * gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    methods = [add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise]
    shuffle(methods)
    noisy_img = img.copy()

    for i in range(randint(1, 3)):
        noisy_img = methods[i](noisy_img)

    return noisy_img

def change_scale(img_path):
    # load image from image path, change the scale, return scaled image
    img = cv2.imread(img_path)

    # Using distortion minimizing shrink algorithm... may be better to do fast resize for more realistic input
    x2 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    x4 = cv2.resize(x2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    x8 = cv2.resize(x4, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    x16 = cv2.resize(x8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    return {"x2": x2, "x4": x4, "x8": x8, "x16": x16}

def remove_color(img_path):
    # load image from image path, remove color, return greyscale image
    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            gray = 0.2989 * img[i, j][2] + 0.5870 * img[i, j][1] + 0.1140 * img[i, j][0]
            img[i, j] = [gray, gray, gray]

    return img

def display_image(img, path=False):
    if path:
        img = cv2.imread(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def get_data_insights():
    from pathlib import Path
    data_path = Path(Path.cwd(), "data")
    insights = {}
    for i in range(1, 900):
        img = cv2.imread(Path(data_path, "truth", f"{i:04}.png"))
        h, w = img.shape[:2]
        insights[(w, h)] = insights.get((w, h), 0) + 1
    print(max(insights, key=insights.get))
    return insights


if __name__ == "__main__":
    from pathlib import Path
    data_path = Path(Path.cwd(), "data")

    for path in Path(data_path, "truth").glob("*.png"):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        if w != 1356 or h != 2040:
            path.unlink()
            continue
        filename, extension = path.parts[-1].split(".")

        # damage
        damaged_path = Path(data_path, "damaged", f"{filename}_d.{extension}")
        cv2.imwrite(damaged_path, add_damage(path))

        # grey
        grey_path = Path(data_path, "grey", f"{filename}_g.{extension}")
        cv2.imwrite(grey_path, remove_color(path))

        # noise
        noisy_path = Path(data_path, "noisy", f"{filename}_n.{extension}")
        cv2.imwrite(noisy_path, add_noise(path))

        # scale
        scaled_images = change_scale(path)
        for scale_factor in scaled_images.keys():
            scale_path = Path(data_path, "scaled", f"{filename}_{scale_factor}.{extension}")
            cv2.imwrite(scale_path, scaled_images[scale_factor])
    
    print("finished")