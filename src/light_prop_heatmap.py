from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

tf.compat.v1.disable_eager_execution()
# I usually do not like referencing files relative to the running script but this is for just two images
_FILE_PATH = Path(__file__).parent.absolute()


# this is from Chollet Jupyter Notebooks Section 5.4
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/660498db01c0ad1368b9570568d5df473b9dc8dd/first_edition/5.4-visualizing-what-convnets-learn.ipynb
def main():
    # Heat map of class activation
    # The local path to our target image
    # This is similar to the example in Section 5.4 from the Chollet book
    img_names = ['artemis.jpg', 'freya.jpg']
    model = load_model('./archive/model_cnn.h5')

    for img_name in img_names:
        img_path = str(_FILE_PATH / img_name)

        # `img` is a PIL image of size 224x224
        img = image.load_img(img_path, target_size=(224, 224))

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x / 255.)

        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        x = preprocess_input(x)

        preds = model.predict(x)
        print(f'Prediction classes for {img_name}:', decode_predictions(preds, top=3)[0])

        predicted_idx = np.argmax(preds[0])

        # This is the entry in the prediction vector with highest probability
        cat_output = model.output[:, predicted_idx]

        # The is the output feature map of the `block5_conv3` layer,
        # the last convolutional layer in VGG16
        last_conv_layer = model.get_layer('block5_conv3')

        # This is the gradient of the most likely class with regard to
        # the output feature map of `block5_conv3`
        grads = K.gradients(cat_output, last_conv_layer.output)[0]

        # This is a vector of shape (512,), where each entry
        # is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of `block5_conv3`,
        # given a sample image
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

        # These are the values of these two quantities, as Numpy arrays,
        # given our sample image of two elephants
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the most likely class
        for i in range(last_conv_layer.filters):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap: np.ndarray = np.mean(conv_layer_output_value, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap / 255.)

        # We use cv2 to load the original image
        img = cv2.imread(img_path)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img

        # for some reason the plot shows the colors backwards but the saved image does not. weird
        plt.subplot(1, 3, 3)
        # flip the BGR of cv2 to the normal RGB
        plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        # Save the image to disk
        cv2.imwrite(str(_FILE_PATH / f"heatmap_{img_name}"), superimposed_img)

        plt.show()


if __name__ == "__main__":
    main()
