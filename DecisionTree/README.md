# Decision Tree Implementation
The ideas for this tree are going to be mainly derived from: https://www.geeksforgeeks.org/decision-tree-implementation-python/

# Packages:
1. sklearn: heavy ml package
2. NumPy: numeric python module
3. Pandas: data file manipulation and io

# Pseudocode:
1. Find the best attribute and place it on the root node of the tree
2. Now, split the training set of the data into subsets. While making the subset make sure that each subset of training dataset should have the same value for an attribute
3. Repeat 1 and 2

# Two main phases in the build
1. Building Phase
  1. Preprocess the dataset
  2. Split the dataset and train using sklearn
  3. Train the classifier
1. Operational Phase
  1. Make predictions
  2. Calculate accuracy
