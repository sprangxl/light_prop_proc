from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

tf.compat.v1.disable_eager_execution()

FILE_PATH = Path(__file__).parent.absolute()

def main():
    # Heat map of class activation
    # The local path to our target image
    # This is similar to the example in Section 5.4 from the Chollet book
    data_names = ['one', 'two']
    model = load_model('./archive/model_crnn.h5')

    crnn_heatmap(data_names, model)


def cnn_heatmap(data_names, model):
    for data_name in data_names:
        data_path = str(FILE_PATH / './data/' / f'{data_name}.npy')

        # import data and scale
        data = (np.squeeze(np.load(data_path)) / 128.) - 1

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = np.sum(data, 0) / 32.
        img = np.repeat(np.expand_dims((x+1)/2, 2), 3, axis=2)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('Grayscale Image')
        plt.imshow(img)

        # We add a dimension to transform our array into a "batch"
        shp = np.shape(x)
        x = np.reshape(x, (1, shp[0], shp[1], 1))

        preds = model.predict(x)
        plt.suptitle(f'Prediction for {data_name}:\n{preds}')

        predicted_idx = np.argmax(preds[0])

        # This is the entry in the prediction vector with highest probability
        cat_output = model.output[:, predicted_idx]

        # The is the output feature map of the `block5_conv3` layer,
        # the last convolutional layer in VGG16
        last_conv_layer = model.layers[1]

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
        plt.title('Heatmap')
        plt.imshow(heatmap / 255., cmap='jet')

        # turn into black and white image
        img = np.load(data_path)
        img = np.repeat(np.expand_dims(np.squeeze(np.sum(img, 0)), 2), 3, axis=2)
        img = np.uint(255. * img / np.max(np.max(img)))

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = (heatmap * 0.4) + (img * 0.6)

        plt.subplot(1, 3, 3)
        # flip the BGR of cv2 to the normal RGB
        plt.title('Heatmap Overlay')
        plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        plt.show()

        # Save the image to disk
        cv2.imwrite(str( f"heatmap_{data_name}_cnn.png"), superimposed_img)

def crnn_heatmap(data_names, model):
    for data_name in data_names:
        data_path = str(FILE_PATH / './data/' / f'{data_name}.npy')

        # import data and scale
        data = (np.squeeze(np.load(data_path)) / 128.) - 1

        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = data
        img = np.repeat(np.expand_dims((x+1)/2, 3), 3, axis=2)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('Grayscale Image')
        plt.imshow(img[0, :, :, :])

        # We add a dimension to transform our array into a "batch"
        shp = np.shape(x)
        x = np.reshape(x, (1, 32, shp[1], shp[2], 1))

        preds = model.predict(x)
        plt.suptitle(f'Prediction for {data_name}:\n{preds}')

        predicted_idx = np.argmax(preds[0])

        # This is the entry in the prediction vector with highest probability
        cat_output = model.output[:, predicted_idx]

        # The is the output feature map of the chosen layer,
        # the last convolutional layer in model
        last_conv_layer = model.layers[1]

        # This is the gradient of the most likely class with regard to
        # the output feature map of chosen layer
        grads = K.gradients(cat_output, last_conv_layer.output)[0]

        # This is a vector where each entry
        # is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of chosen layer,
        # given a sample image
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

        # These are the values of these two quantities, as Numpy arrays,
        # given our sample image
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
        plt.title('Heatmap')
        plt.imshow(heatmap / 255., cmap='jet')

        # turn into black and white image
        img = np.load(data_path)
        img = np.repeat(np.expand_dims(np.squeeze(np.sum(img, 0)), 2), 3, axis=2)
        img = np.uint(255. * img / np.max(np.max(img)))

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = (heatmap * 0.4) + (img * 0.6)

        plt.subplot(1, 3, 3)
        # flip the BGR of cv2 to the normal RGB
        plt.title('Heatmap Overlay')
        plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        plt.show()

        # Save the image to disk
        cv2.imwrite(str( f"heatmap_{data_name}_cnn.png"), superimposed_img)


if __name__ == "__main__":
    main()
