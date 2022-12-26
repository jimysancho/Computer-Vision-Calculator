# Computer Vision Calculator

## 1. Requirements

In order for this application to work, the photo must have the next **format**: 

1. A big square / rectangle must surround the equations. 
2. The equations have to be visually separated from one another. 
3. The characters from each equation have to be visually separated from one another. 
4. The format of the equations must be: number, letter, algebraic symbol. Even if the number is a 0. 

![Getting Started](./images/equation.png)

## 2. Process

Using the `data/perspective.py` file, we want to get the block equation in order to identify the character's equations. 
Here is the process: 

### 2.1 Segmentation of the block

Using some computer vision techniques (such as Otsu segmenation, dilation and erosion) we get the area in which the equations are written. That is why the equations have to be surrounded by a big box. 

![Getting Started](./images/mask.png)

### 2.2 Vertices

If the image is rotated, we need to obtain the vertices of the big box to tranform the perspective: 

![Getting Started](./images/vertices.png)


### 2.3 Getting the individual equations

Using only erosion and dilation over the area we got in the section 2.1, we obtain the individual equations: 

![Getting Started](./images/block_equations.png)

### 2.4 Single Characters

Once we have each equation (and its correspondent coordinates), we can again apply erosion and dilation over that equation's area to obtain the single characters that then will be fed to the neural network: 

![Getting Started](./images/single_equation.png)