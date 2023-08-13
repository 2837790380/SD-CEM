# Learning Hierarchy-Enhanced POI Category Representations Using Disentangled Mobility Sequences

These are the source codes for the **SD-CEM** model and its corresponding data.

- Data
  1. dataset\data\category.csv — POI category data, including category ID, category name, and the hierarchical parent nodes
  2. dataset\data\CheckinCategoryIDSequenceJP.csv — A dataset of the JP mobility sequences as input, containing 10,336 sequences of POI categories and encompassing 330 distinct categories.
  3. dataset\data\CheckinCategoryIDSequenceUS.csv — A dataset of the US mobility sequences as input, containing 21,898 sequences of POI categories and encompassing 398 distinct categories.
- Code
  1. main.py — A python file to run the SD-CEM model. Noted that there are some hyperparameter that   can be set in this file.
  2. trainer — The files in this folder contain the details of model training and optimization.
  3. model — These are the codes for **SD-CEM**, including the implementation details of all components of the model.
- embeddings — These are the trained category embeddings. The named format is [SD-CEM#dataset#embedding size.csv], and the data format is [category name, category embedding]

- tasks
  1. tasks\matchrate.py — This file aims to calculate the match rate based on the category embeddings.
  2. tasks\mobility.py — A file to run to get the performance on the mobility task.
  3. tasks\recommendation.py — A file to run to get the performance on the POI recommendation task.
