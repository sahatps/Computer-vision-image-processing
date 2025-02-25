# Iterative Refinement Triangulation for 3D Reconstruction of Reach-to-Grasp Movements Using Stereo Cameras

This project presents a method for accurately reconstructing 3D trajectories of reach-to-grasp movements by employing stereo camera setups and iterative refinement triangulation techniques. The approach aims to enhance the precision of 3D motion capture in applications such as robotics, biomechanics, and human-computer interaction.

---

## Project Overview

Accurate 3D reconstruction of hand movements is crucial for analyzing reach-to-grasp actions. This project utilizes stereo vision systems to capture synchronized images from two viewpoints, enabling depth perception through triangulation. By applying iterative refinement to the triangulation process, the method reduces errors and improves the fidelity of the reconstructed 3D trajectories.

---

## Input, Output, Objective, and Limitations

1. **Input:**
   - Synchronized stereo image pairs capturing reach-to-grasp movements.
   - Calibration data for both cameras, including intrinsic and extrinsic parameters.

2. **Output:**
   - High-precision 3D coordinates representing the trajectory of hand movements.
   - Visualization of the reconstructed 3D path.

3. **Objective:**
   - To develop a robust system that accurately reconstructs 3D hand trajectories from stereo images.
   - To enhance the precision of 3D motion capture through iterative refinement of triangulated data.
   - To provide a tool for detailed analysis of reach-to-grasp movements in various applications.

4. **Limitations:**
   - Requires precise camera calibration; inaccuracies can lead to reconstruction errors.
   - Dependent on the quality and synchronization of stereo image capture.
   - Computational complexity increases with the number of iterations in the refinement process.
   - Performance may be affected by occlusions or poor lighting conditions during image capture.

---

## Comparison of Triangulation Methods

| **Step** | **MATLAB's `triangulate` Function** | **Custom `myTriangulate` Function** |
|----------|------------------------------------|------------------------------------|
| **1. Input Data Preparation** | - **Inputs:**<br>  Two sets of corresponding 2D points (`matchedPoints1`, `matchedPoints2`) and either stereo parameters (`stereoParams`) or individual camera projection matrices (`cameraMatrix1`, `cameraMatrix2`).<br>- **Preprocessing:**<br>  Undistort the image points if using stereo parameters. | - **Inputs:**<br>  Similar sets of corresponding 2D points and camera parameters.<br>- **Preprocessing:**<br>  May include additional steps such as outlier detection or normalization of points to improve numerical stability. |
| **2. Initial Triangulation Estimate** | - **Method:**<br>  Performs linear triangulation using the Direct Linear Transformation (DLT) approach, solving a system of linear equations to find the 3D point that best fits the observed 2D correspondences. | - **Method:**<br>  May start with a similar linear triangulation to obtain an initial estimate. |
| **3. Iterative Refinement** | - **Approach:**<br>  Typically does not include iterative refinement; it provides a direct solution based on the initial linear estimate. | - **Approach:**<br>  May implement iterative methods to refine the initial 3D point estimate.<br>- **Techniques:**<br>  Methods such as minimizing reprojection error using optimization techniques. |
| **4. Outlier Handling** | - **Capability:**<br>  Does not inherently handle outliers; assumes that all input correspondences are valid. | - **Capability:**<br>  May include mechanisms to detect and exclude outliers, such as RANSAC (Random Sample Consensus), enhancing robustness to mismatched points. |
| **5. Output** | - **Results:**<br>  Returns the 3D coordinates of points and optionally, reprojection errors and validity indices indicating points located in front of both cameras. | - **Results:**<br>  Likely returns refined 3D coordinates, possibly with additional information such as confidence metrics or detailed error statistics. |

---

## How to Use

1. **Setup:**
   - Arrange two cameras in a stereo configuration, ensuring overlapping fields of view covering the workspace.
   - Calibrate the stereo camera system to obtain intrinsic and extrinsic parameters.

2. **Data Acquisition:**
   - Capture synchronized stereo image pairs of the subject performing reach-to-grasp movements.
   - Ensure consistent lighting and minimal background clutter to improve image quality.

3. **3D Reconstruction Process:**
   - **Feature Detection:** Identify and match key points (e.g., fingertips) in both images using feature detection algorithms.
   - **Initial Triangulation:** Compute initial 3D coordinates from matched points via linear triangulation.
   - **Iterative Refinement:** Apply optimization techniques to minimize reprojection errors, refining the 3D coordinates iteratively.
   - **Visualization:** Render the reconstructed 3D trajectory for analysis and interpretation.

4. **Experimentation:**
   - Adjust camera positions and calibration settings to optimize reconstruction accuracy.
   - Experiment with different feature detection and matching algorithms to enhance robustness.
   - Modify iterative refinement parameters to balance computational load and precision.

---

## Contributing

Contributions to improve the system's accuracy, efficiency, and applicability are welcome. Please submit issues or pull requests to the project's repository for discussion and integration.

---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: This project builds upon established methods in stereo vision and 3D reconstruction. For foundational concepts and algorithms, refer to resources such as the [OpenCV documentation on camera calibration and 3D reconstruction](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html) and relevant literature in computer vision.*
