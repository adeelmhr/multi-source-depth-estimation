import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, LeakyReLU

def create_model():
    print('Loading base model (EfficientNetB0)..')

    # Encoder Layers
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(480, 640, 3))
    print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    # Layer freezing?
    for layer in base_model.layers:
        layer.trainable = True

    # Starting number of decoder filters
    decode_filters = int(base_model_output_shape[-1] // 2)
    base_model.summary()

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = UpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
        up_i = Concatenate(name=name + '_concat')([up_i, base_model.get_layer(concat_with).output])  # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder
    up1 = upproject(base_model.output, decode_filters, 'up1', 'block6a_expand_activation')
    up2 = upproject(up1, decode_filters // 2, 'up2', 'block4a_expand_activation')
    up3 = upproject(up2, decode_filters // 4, 'up3', 'block3a_expand_activation')
    up4 = upproject(up3, decode_filters // 8, 'up4', 'block2a_expand_activation')

    # Final convolution
    conv_final = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv_final')(up4)

    model = Model(inputs=base_model.inputs, outputs=conv_final)
    print('\\nModel created.')
    return model
