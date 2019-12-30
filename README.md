# Cardiac_cycle_feature_learning_architecture
Left ventricular ejection fraction (LVEF) is of significant importance for early identification and diagnosis of cardiac disease, but the estimation of LVEF with consistently reliable and high accuracy is still a great challenge due to the high variability of cardiac structures and the complexity of temporal dynamics in cardiac MRI sequences. The widespread methods of LVEF estimation rely on the left ventricular volume. Thus strong prior knowledge is often necessary, which impedes the ease of use of existing methods as clinical tools. In this paper, we propose a cardiac cycle feature learning architecture to achieve an accurate and reliable estimation of LVEF. Unlike the segmentation-based methods, this architecture uses the direct estimation method and does not rely on strong prior knowledge. Experiments on 2900 left ventricle segments of 145 subjects from short axis MR sequences of multiple lengths prove that our proposed method achieves reliable performance (Correlation Coefficient: 0.946; Mean Absolute Error 2.67; Standard Deviation: 3.23). As the first solution to directly estimate LVEF, our proposed method demonstrates great potential in future clinical applications.

Usage:

ef_estimate_flow.m: An example of estimating optical flow and calculating cardiac cycles
motion_feature_fusion_extraction_regression: Our deep learning frame.

Acknowledgment:

Thanks to Z Han, Sun D, T. A. Davis, Y. Rubner, and M. A. Ruzon for their public source codes.

References:

Sun, D.; Roth, S. & Black, M. J. "Secrets of Optical Flow Estimation and Their Principles" IEEE Int. Conf. on Comp. Vision & Pattern Recognition, 2010

Sun, D.; Roth, S. & Black, M. J. "A Quantitative Analysis of Current Practices in Optical Flow Estimation and The Principles Behind Them" Technical Report Brown"CS"10"03,
2010
