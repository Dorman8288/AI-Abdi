# Assignments for Artificial Inteligence Course 
this repository features 5 diffrent projects form diffrent areas of artificial inteligence that I did at my AI course in IUST University. a brief explanation for every project is provided in the following.
the statements for each Project is Provided in the **Statements** folder, altough they are in Persian. I also wrote a report for some of the projects for furthur explanations and details.

# Project 1 - Decision Trees
## Task Specification
given data from airlines determine whether a customer is happy with their flight or not.

i used and implemented Decision Trees for solving this classification problem. i experimented with ways of data spliting, discretizing data and optimizing hyperparameters. there exist a more detailed report of my soloution in main.ipynb.
## Project Highlights
### Hyperparameter Tuning
i used prepruning for building the tree until a certain depth to avoid overfitting. this naturaly introduces hyperparameter d(maximum depth of the tree).
one of the challenges of this task was the choice of d(Depth of the constructed tree). this is a chart for training and validation accuracy. you can visibly see where the overfitting occurs. so the best depth for our model is around 10.

![Accuracy](/P1/stats/DepthAnalysis.png)

*orange line - test accuracy*

*blue dotted line - train accuracy*

### Discretizing 
