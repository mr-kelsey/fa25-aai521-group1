import cv2
import matplotlib.pyplot as plt
import numpy as np

class Completion:
    def __init__(self, image, pipeline, input_shape):
        self.x_hops = None
        patches = self.get_patches(image, input_shape)
        processed = [pipeline(patch, attention_mask) for patch, attention_mask in patches]

        generated_image = self.stitch_image(processed)
        self.visualize(image, generated_image)

    def get_patches(self, image, input_shape):
        height, width = image.shape[:2]
        h_step, w_step = input_shape

        ideal_height = (height // h_step + 1) * h_step
        ideal_width = (width // w_step + 1) * w_step
        padded_image = cv2.copyMakeBorder(image, 0, ideal_height - height, 0, ideal_width - width, cv2.BORDER_CONSTANT)
        attention_mask = cv2.copyMakeBorder(np.ones((height, width)), 0, ideal_height - height, 0, ideal_width - width, cv2.BORDER_CONSTANT)

        h_indicies = [i * h_step for i in range(height // h_step + 1)]
        w_indicies = [i * w_step for i in range(width // w_step + 1)]
        self.x_hops = len(w_indicies)

        out = []
        for coords in  [(h_indicies[i], w_indicies[j]) for i in range(len(h_indicies)) for j in range(len(w_indicies))]:
            x, y = coords
            image_patch = padded_image[x:x + w_step, y:y + h_step]
            mask_patch = attention_mask[x:x + w_step, y:y + h_step]
            out.append((image_patch, mask_patch))

        return out

    def stitch_image(self, image_parts):
        patch_pieces = []
        mask_pieces = []

        for p_slice in [image_parts[i:i + self.x_hops] for i in range(0, len(image_parts), self.x_hops)]:
            patch_row = []
            mask_row = []
            for patch, mask in p_slice:
                patch_row.append(patch)
                mask_row.append(mask)
            patch_pieces.append(np.concat(patch_row, axis=1))
            mask_pieces.append(np.concat(mask_row, axis=1))


        img = np.concat(patch_pieces)
        mask = np.concat(mask_pieces)

        crop_height = len(mask[mask[:, 0] >= 0.5])
        crop_width = np.where(mask[0] <= 0.5)[0][0]

        return img[:crop_height, :crop_width]

    def visualize(self, original, processed):
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(original)
        plt.title("original")
        fig.add_subplot(122)
        plt.imshow(processed)
        plt.title("processed")
        plt.show()

if __name__ == "__main__":
    def pipeline(patch, attention_mask):
        patch = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        attention_mask = cv2.resize(attention_mask, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        attention_mask = np.clip(attention_mask, 0, 1)
        return patch, attention_mask
    
    image = cv2.imread("/home/poppop/class/Week 7/fa25-aai521-group1/data/scaled/0006_x16.png")
    input_shape = (100, 100)
    output_shape = (300, 300)

    completion = Completion(image, pipeline, input_shape)