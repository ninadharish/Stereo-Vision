# Stereo Vision

## Description

Given a set of 2 images of the same object from two different camera angles, this project attempts to draw the disparity and depth map.


## Data

* Images of the same object from different camera angles

![alt text](/data/octagon/im0.png)

![alt text](/data/octagon/im1.png)

* Camera Parameters


## Approach

* Feature matching using Orb Feature Matching

* Compute the Fundamental Matrix using point correspondences (Least Squares as well as RANSAC approach)

* Compute the Essential Matrix from the Fundamental Matrix

* Decompose the Essential Matrix to get the Rotation and Translation matrices

* Find the Perspective Transforms of the two images with respect to each other

* Warp the two images based on these transforms for Rectification (Vectorized implementation)

* Compute the Epipolar Lines for these rectified images based on the point correspondences

* Using a sliding window approach, found the minimum of the Sum of Squared Distance for a specified window size in a single row and found window correspondences.

* The disparity was found using thje difference in the pixel locations.

* The differences were sclaed between 0 - 255 and this was plotted on a grayscale image. Heatmap conversion of the disparity map was also displayed.

* Based on the triangulation formula, the depth of the points was found and a similar grayscale was well as heat map was created.


## Output

* Feature Matching

![alt text](/output/out1.jpg)

* Fundamental and Essential Matrix

![alt text](/output/out2.jpg)

* Rotation Matrices

![alt text](/output/out3.jpg)

* Translation and Homography Matrices

![alt text](/output/out4.jpg)

* Epipolar Lines on rectified images

![alt text](/output/out5.jpg)

* Disparity Map (Grayscale and Heatmap)

![alt text](/output/out6.jpg)

![alt text](/output/out7.jpg)

* Depth Map (Grayscale and Heatmap)

![alt text](/output/out8.jpg)

![alt text](/output/out9.jpg)



## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/Stereo-Vision.git
```

* Open the repository and navigate to the `src` folder.
```
cd Stereo-Vision/src
```
* Depending on whether you want to superimpose athe image or 3D cube on the tag, comment/uncomment the proper line.

* Run the program.
```
python main.py
```


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)