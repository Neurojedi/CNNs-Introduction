import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
def preview_dataset(dataset,get_label_name):
    plt.figure(figsize=(12, 12))
    plot_index = 0
    for features in dataset.take(12):
        (image, label) = features
        plot_index += 1
        plt.subplot(3, 4, plot_index)
        # plt.axis('Off')
        label = get_label_name(label.numpy())
        plt.title('Label: %s' % label)
        plt.imshow(image.numpy())

INPUT_IMG_SIZE = 150
def format_example(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Make sure that image has a right size
    image = tf.image.resize(image, [INPUT_IMG_SIZE, INPUT_IMG_SIZE])
    return image, label

IMAGE_RES = 224
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


def plot_vawes():
    # Generate x values from 0 to 2*pi
    x = np.linspace(0, 2*np.pi, 100)

    # Compute the corresponding y values for the sine wave
    y_sin = np.sin(x)

    # Apply ReLU to the y values of the sine wave
    y_relu = np.maximum(0, y_sin)

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the sine wave in the first subplot
    ax1.plot(x, y_sin)
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')
    ax1.set_title('Sine Wave')
    ax1.grid(True)

    # Plot the ReLU(applied to the sine wave) in the second subplot
    ax2.plot(x, y_relu)
    ax2.set_xlabel('x')
    ax2.set_ylabel('ReLU(sin(x))')
    ax2.set_title('ReLU Applied to Sine Wave')
    ax2.grid(True)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
