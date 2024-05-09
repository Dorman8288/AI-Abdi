# Assignments for Artificial Inteligence Course 
This repository showcases five distinct projects covering various areas of artificial intelligence that I completed during my studies at IUST University. A brief overview of each project is provided below.

The statements for each project are available in the Statements folder, though they are in Persian. Additionally, I have written reports for some of the projects to provide further explanations and details.

# Project 1 - Decision Trees
## Task Specification
The task involves analyzing data from airlines to determine customer satisfaction with their flights.

I utilized and implemented Decision Trees to address this classification problem. i experimented with ways of data spliting, Discretization data and optimizing hyperparameters. A comprehensive report detailing my solution can be found in main.ipynb.
## Project Highlights
### Hyperparameter Tuning
To prevent overfitting, I employed pre-pruning by limiting the tree depth to a certain level. This naturally introduced a hyperparameter, 'd' (maximum depth of the tree). Determining the appropriate depth was challenging, illustrated in the following chart showcasing training and validation accuracy. Optimal depth, approximately 10, was identified to mitigate overfitting.

![Accuracy](/P1/stats/DepthAnalysis.png)

*orange line - test accuracy*

*blue dotted line - train accuracy*

### Discretization 
Given the dataset's continuous fields, data discretization was crucial during development. I opted for an approach that involved dividing the dataset into an arbitrary number of buckets (k) to ensure homogeneous data splitting. This method categorizes data points uniformly across buckets, resulting in higher categorization levels for more common data points and fewer categories for outliers.

![Accuracy](/P1/stats/data_split_5_percent.png)

*This image illustrates data points and their respective categories.*

#Project 2 - Neural Networks
##Task Specification
the point of this project was familirization with basic neural networks and machine learning concepts. this project consists of 6 tasks that are discussed below. for implementation i chose the pytorch library.

### Task 1
preform the privious project task with neural networks.

### soloution 
I made use of a simple feed forward neural network and was able to achive 99% test and training accuracy in just few epochs of data.

### Task 2
given a number of datapoints from a arbitarary mathematic function you should predict the function.

### soloution 
i used a
