<h1 align="center">
  Multi-Source Depth Estimation: Utilizing Real, Synthetic, and Monocular Depth Data with Custom Loss Functions
</h1>

<p align="center">
  This project presents a progressive fine-tuning approach for monocular metric depth estimation using data from multiple sources, including real RGB-D data, synthetic scenes, and pseudo-labelled monocular images. The method combines DenseNet169 and EfficientNet-B0 encoder–decoder architectures with a multi-scale loss incorporating MAE, edge, SSIM, and perceptual components to improve depth accuracy and cross-dataset generalisation.
</p>

## Qualitative Results

The following figure presents sample depth predictions produced by the trained models across different indoor scenes. The results demonstrate the models' ability to recover the overall scene structure, object boundaries, and depth variations from a single RGB image.

<p align="center">
  <img src="results_1.png"
       alt="Qualitative results of the multi-source depth-estimation models"
       width="800">
</p>

<p align="center">
  <em>Example monocular depth predictions generated using the proposed multi-source progressive fine-tuning approach.</em>
</p>
