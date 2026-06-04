# Machine Learning Lifecycle

The Machine Learning lifecycle is the complete journey of building, deploying, and maintaining an ML model. <br>
It covers everything from collecting raw data to monitoring the model in production. <br>

Below is a clear step-by-step breakdown:

### Problem Definition
Understand what problem you are solving. <br>
Define the goal: 
- classification, 
- regression, 
- clustering, etc.

Identify what a good outcome looks like (accuracy, latency, cost). <br>

### Data Collection

Gather raw data from various sources: 
- databases, 
- APIs, 
- logs,
- sensors,
- user inputs, etc.

Ensure the data represents real-world scenarios. <br>

### Data Cleaning & Preparation

Handle missing values.
- Remove noise and duplicates.
- Fix incorrect or inconsistent entries.
- Split the data into train/validation/test sets.

> This is usually the most time-consuming step.

### Feature Engineering

Transform raw data into meaningful features.
Examples: 
- converting timestamps, 
- extracting text embeddings, 
- scaling numbers, 
- one-hot encoding categories.

Better features → better model performance.

### Model Selection

Choose the right algorithm based on the problem and data:
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Neural Networks
etc.

### Model Training

Feed training data into the algorithm.

The model learns patterns and relationships.

Adjust parameters to minimize the error.

### Model Evaluation

Test model performance on validation/test datasets.

Common metrics: `Accuracy`, `F1-score`,`RMSE`, `ROC-AUC`.

Check if the model meets the defined success criteria.

### Hyperparameter Tuning

Improve performance using techniques like Grid Search, Random Search, Bayesian Optimization.

Examples: learning rate, depth of trees, regularization values.

### Model Deployment

Move the model from development to production. <br>
Deployment options:
- REST API
- Batch jobs
- Edge devices
- Cloud ML services (SageMaker, Vertex AI, etc.)

### Monitoring & Logging

- Monitor model accuracy, drift, latency, errors.
- Track data distribution and real-world performance.
- Set alerts for anomalies or performance drops.

### Model Maintenance

Retrain with new data. <br>

Update the pipeline when business logic or data changes. <br>

Version control for data, code, and models. <br>
