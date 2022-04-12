from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

# disable eager execution to allow for analysis of gradients
tf.compat.v1.disable_eager_execution()
# get path from current directory
FILE_PATH = Path(__file__).parent.absolute()

def main():
    # Heat map of class activation the local path to our target image
    # This is similar to the example in Section 5.4 from the Chollet book
    # Use archived model
    data_names = ['zero', 'one', 'two']
    model = load_model('./archive/model_cnn.h5')
    cnn_heatmap(data_names, model)


def cnn_heatmap(data_names, model):
    for data_name in data_names:
        # get path of data of interest
        data_path = str(FILE_PATH / './data/' / f'{data_name}.npy')

        # import data and scale
        data = (np.squeeze(np.load(data_path)) / 128.) - 1

        # `x` is a float32 Numpy array of shape (32, 140, 146)
        x = np.sum(data, 0) / 32.
        img = np.repeat(np.expand_dims((x+1)/2, 2), 3, axis=2)

        # plot original image as reference
        fig = plt.figure()
        plt.subplot(1, 4, 1)
        plt.title('Grayscale Image')
        plt.imshow(img)
        plt.axis('off')

        # We add a dimension to transform our array into a "batch"
        shp = np.shape(x)
        x = np.reshape(x, (1, shp[0], shp[1], 1))

        # make prediction from data using model
        preds = model.predict(x)
        plt.suptitle(f'Prediction for {data_name}:\n{preds}')

        for ii in range(3):
            # This is the entry in the prediction vector for given result
            cat_output = model.output[:, ii]

            # The is the output feature map of the layer, the last convolutional layer
            last_conv_layer = model.layers[1]

            # This is the gradient of the most likely class with regard to the output feature map of this layer
            grads = K.gradients(cat_output, last_conv_layer.output)[0]

            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

            # This function allows us to access the values of the quantities we just defined:
            # `pooled_grads` and the output feature map of this layer, given a sample image
            iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

            # These are the values of these two quantities, as Numpy arrays, given our sample image
            pooled_grads_value, conv_layer_output_value = iterate([x])

            # We multiply each channel in the feature map array
            # by how important this channel is with regard to the most likely class
            for i in range(last_conv_layer.filters):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

            # The channel-wise mean of the resulting feature map is our heatmap of class activation
            heatmap: np.ndarray = np.mean(conv_layer_output_value, axis=-1)
            min_h = np.min(heatmap)
            max_h = np.max(heatmap)
            heatmap = 2 * ((heatmap - min_h) / (max_h - min_h)) - 1
            heatmap = np.maximum(heatmap, 0)

            # We resize the heatmap to have the same size as the original image
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            # plot
            plt.subplot(2, 4, 2 + ii)
            plt.axis('off')
            plt.title(f'Heatmap {ii}')
            plt.imshow(heatmap, cmap='inferno')

            # turn into black and white image
            img = np.load(data_path)
            img = np.repeat(np.expand_dims(np.squeeze(np.sum(img, 0)), 2), 3, axis=2)
            img = img / np.max(np.max(img))
            img = np.uint(255 * img)

            # We convert the heatmap to RGB
            heatmap = np.uint8(255 * heatmap)

            # We apply the heatmap to the original image
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

            # 0.4 here is a heatmap intensity factor
            superimposed_img = (heatmap * 0.4) + (img * .8)

            # plot heatmap overlayed on data
            plt.subplot(2, 4, 6 + ii)
            plt.axis('off')
            plt.title(f'Heatmap\nOverlay {ii}')
            # flip the BGR of cv2 to the normal RGB
            plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        plt.tight_layout()
        plt.savefig(str(f"heatmap_{data_name}_cnn.png"))
        plt.show()

        # Save the image to disk


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

        # output feature map of the chosen layer, the last convolutional layer in model
        last_conv_layer = model.layers[1]

        # Gradient of the most likely class with regard to the output feature map of chosen layer
        grads = K.gradients(cat_output, last_conv_layer.output)[0]

        # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of chosen layer, given a sample image
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

        # These are the values of these two quantities, as Numpy arrays, given the sample image
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # multiply each channel in the feature map array by how important this channel is
        # with regard to the most likely class
        for i in range(last_conv_layer.filters):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # The channel-wise mean of the resulting feature map is our heatmap of class activation
        heatmap: np.ndarray = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)

        plt.subplot(1, 3, 2)
        plt.title('Heatmap')
        plt.imshow(heatmap, cmap='jet')

        # turn into black and white image
        img = np.load(data_path)
        img = np.repeat(np.expand_dims(np.squeeze(float(np.sum(img, 0))), 2), 3, axis=2)
        img = img / np.max(np.max(img))

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # superimpose heatmap onto image
        superimposed_img = (heatmap * 0.5) + (img * 0.5)

        plt.subplot(1, 3, 3)
        # flip the BGR of cv2 to the normal RGB
        plt.title(f'Heatmap Overlay')
        plt.imshow(superimposed_img[..., [2, 1, 0]] / 255.)

        plt.show()

        # Save the image to disk
        cv2.imwrite(str( f"heatmap_{data_name}_cnn.png"), superimposed_img)


if __name__ == "__main__":
    main()
