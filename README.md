# Image Color Segmentation

## Overview
This project, by Kaifan Zheng and Noshin Chowdhury, introduces an efficient image segmentation method using Principal Component Analysis (PCA) to optimize Gaussian Mixture Models (GMM) and Expectation-Maximization (EM) Algorithm.

## Key Concepts
- **PCA**: A technique for data structure exploration and dimensionality reduction.
- **K-Means Clustering**: A simple clustering algorithm for local optima computation.
- **GMM and EM Algorithm**: Used for estimating optimal parameters and clustering discrete data.

## Implementation
- The combination of PCA with GMM and EM reduces segmentation time significantly (~25x faster) while maintaining high accuracy.
- The approach is particularly effective for large image datasets.

## Results
- Experiments show improved speed and accuracy in image segmentation tasks.
- Test results demonstrate efficient mean color segmentation and accurate color averaging for smoother output images.

**Labled Image , k = 7**
<p align=center>
    <img src="https://github.com/kaifanzheng/Image-Clustering-based-on-PCA-GMM/blob/main/SCR/resultImg/labeled%20image%20k%3A%207.jpeg">
</p>

**averaged Image , k = 7**
<p align=center>
    <img src="https://github.com/kaifanzheng/Image-Clustering-based-on-PCA-GMM/blob/main/SCR/resultImg/average%20image%20k%3A%207.jpeg">
</p>

## References
- For detailed mathematical formulations and algorithmic explanations, refer to the cited bibliography in the report.
