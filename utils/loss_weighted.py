import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def depth_loss_function(y_true, y_pred, maxDepthVal=1000.0/10.0, w1=0.5, w2=0.2, w3=1.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self, layer_names, vgg_model=None):
        super(PerceptualLoss, self).__init__()
        if vgg_model is None:
            vgg_model = VGG16(include_top=False, weights='imagenet')
        self.model = Model(inputs=vgg_model.input, outputs=[vgg_model.get_layer(name).output for name in layer_names])
        self.model.trainable = False

    def call(self, y_true, y_pred):
        y_true = tf.image.grayscale_to_rgb(y_true)
        y_pred = tf.image.grayscale_to_rgb(y_pred)
        
        y_true_features = self.model(y_true)
        y_pred_features = self.model(y_pred)
        
        perceptual_loss = 0.0
        for y_true_feature, y_pred_feature in zip(y_true_features, y_pred_features):
            perceptual_loss += tf.reduce_mean(tf.square(y_true_feature - y_pred_feature))
        
        return perceptual_loss

def combined_loss(y_true, y_pred, maxDepthVal=1000.0/10.0, alpha=1.0, beta=0.5, w1=0.5, w2=0.2, w3=1.0):
    # Original depth loss with weighted components
    depth_loss = depth_loss_function(y_true, y_pred, maxDepthVal, w1, w2, w3)
    
    # Perceptual loss
    layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1']
    perceptual_loss_fn = PerceptualLoss(layer_names)
    perc_loss = perceptual_loss_fn(y_true, y_pred)
    
    # Combine losses with weights
    return alpha * depth_loss + beta * perc_loss

# Example usage in model.compile:
# model.compile(optimizer='adam', loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha=1.0, beta=0.5, w1=0.5, w2=0.2, w3=1.0))
