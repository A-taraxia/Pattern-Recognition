# Pattern-Recognition

**Exercise description in English:**

Property Price Estimation Using Machine Learning Algorithms

The objective of this particular exercise is to develop machine learning algorithms for estimating the median value of properties in a broader geographical area of California based on a set of objective property characteristics in that region. Each such area represents the smallest geographical unit recorded in the 1990 census, with a population ranging from 600 to 3000 residents. The relevant dataset is stored in the accompanying file 'housing.csv.'

The training process for the involved machine learning mechanisms should rely on a set of objective property attributes in each geographical area, including the following characteristics:

The geographical longitude of the area's center.
The geographical latitude of the area's center.
The median age of the properties (housing_median_age) in the area.
The total number of rooms (total_rooms) in the area's properties.
The total number of bedrooms (total_bedrooms) in the area's properties.
The population of the area.
The number of households (households) in the area.
The median income (median_income) of the area's residents.
The proximity to the ocean (ocean_proximity) in the area.
The goal is to estimate:

The median value (median_house_value) of the properties in the area.
Data Preprocessing

You should identify subsets of numeric and categorical features.
For the subset of numeric features, you should experiment with different data scaling techniques to represent all numeric features on the same scale.
For the subset of categorical features, you can use One Hot Vector encoding to represent the data vectorially.
You should identify if there are numeric features with missing values. For these records, you can fill in the missing values with the median value of the feature.
Data Visualization

Graphically represent frequency histograms (corresponding to probability density functions) for each of the 10 variables involved in the problem.
Attempt to create two-dimensional data plots in which combinations of 2, 3, or 4 variables are clearly represented.
Data Regression

Implement the Perceptron Algorithm so that the trained learning mechanism implements a linear discriminant function of the form g: R^l → {-1, +1}, where l is the dimension of the final feature space. Consider an appropriate threshold in the property values so that the continuous range of their values is divided into two distinct sets, allowing you to treat the problem as a binary classification problem. In this particular exercise, you cannot use pre-existing functions.
Implement the Least Squares Algorithm so that the trained learning mechanism implements a linear regression of the form g: R^l → R, where l is the dimension of the final feature space. In this specific exercise, you cannot use pre-existing functions.
Implement a multi-layer neural network so that the trained learning mechanism performs a non-linear regression of the form g: R^l → R, where l is the dimension of the final feature space. In this exercise, you can use pre-existing functions.
