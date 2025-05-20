# customer_segmentation.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime # No longer needed for Mall_Customers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

print("--- Starting Customer Segmentation Project (Mall Customers) ---")

# Load the dataset
# Make sure 'Mall_Customers.csv' is in the same directory as your script
file_path = 'Mall_Customers.csv' # <--- Changed filename
try:
    df = pd.read_csv(file_path) # <--- Changed to read_csv
    print(f"\nDataset '{file_path}' loaded successfully!")
    print(f"Initial number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please make sure the file is in the correct directory.")
    print(f"You can download it from Kaggle (search for 'Mall Customer Segmentation Data').")
    exit()

print("\n--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Basic information about the dataset ---")
df.info()

print("\n--- Descriptive statistics of numerical columns ---")
print(df.describe())

# Step 2: Exploratory Data Analysis (EDA) and Feature Selection

# For Mall_Customers, we'll focus on 'Annual Income (k$)' and 'Spending Score (1-100)'
# as our clustering features.

print("\n--- Checking for missing values in the dataset ---")
print(df.isnull().sum())
# This dataset is usually very clean, so you should see 0 missing values.

print("\n--- Visualizing the distribution of key features ---")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age')

plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Distribution of Annual Income (k$)')

plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title('Distribution of Spending Score (1-100)')
plt.tight_layout() # Adjusts plot parameters for a tight layout
plt.show() # Display the plots

print("\n--- Visualizing relationships between key clustering features ---")
# Scatter plot of Annual Income vs Spending Score
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, s=100) # s=size of points
plt.title('Annual Income (k$) vs. Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show() # Display the scatter plot

# Select the features for clustering
# We are explicitly choosing these two numerical columns
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print(f"\nFeatures selected for clustering (X) shape: {X.shape}")
print(X.head()) # Show first few rows of selected features

# Step 3: Feature Scaling

print("\n--- Scaling the features ---")

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the selected features (X) and transform them
# X_scaled will now have a mean of 0 and standard deviation of 1 for each feature
X_scaled = scaler.fit_transform(X)

# Convert the scaled array back to a DataFrame for easier inspection
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Features scaled successfully!")
print(X_scaled_df.head()) # Display first few rows of scaled data
print(X_scaled_df.describe()) # Display descriptive stats of scaled data (means should be near 0, stds near 1)

# Step 4: Determine Optimal Number of Clusters (Elbow Method)

print("\n--- Applying Elbow Method to find optimal K ---")

wcss = [] # Within-Cluster Sum of Squares (Inertia)
# Test K-Means for a range of possible cluster numbers (e.g., from 1 to 10)
for i in range(1, 11):
    # Initialize KMeans with 'k' clusters
    # n_init='auto' handles the initialization algorithm
    # random_state for reproducibility
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(X_scaled) # Fit KMeans on the scaled data
    wcss.append(kmeans.inertia_) # inertia_ is the WCSS value

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11)) # Ensure x-axis ticks are integers
plt.grid(True)
plt.show() # Display the plot

print("Please examine the plot and identify the 'elbow' point.")
print("This point suggests the optimal number of clusters (K) for your data.")

# Step 5: Apply K-Means Clustering

# Assuming K=5 is the optimal number of clusters from the Elbow Method for this dataset
optimal_k = 5
print(f"\n--- Applying K-Means Clustering with K = {optimal_k} ---")

# Initialize KMeans with the optimal number of clusters
# n_init='auto' is good practice for multiple initializations
kmeans_model = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)

# Fit the KMeans model on the scaled data
kmeans_model.fit(X_scaled)

# Get the cluster labels for each data point
# These labels indicate which cluster each customer belongs to (0, 1, 2, 3, 4)
cluster_labels = kmeans_model.labels_

# Add the cluster labels back to our original (unscaled) DataFrame for analysis
df['Cluster'] = cluster_labels

print(f"Clustering complete. Each customer is assigned to one of {optimal_k} clusters.")
print("\n--- First 5 rows with new 'Cluster' column ---")
print(df.head())

# Step 6: Analyze and Interpret Clusters

print("\n--- Analyzing Cluster Characteristics ---")

# Group the original data by 'Cluster' and calculate the mean of our features
cluster_centers_df = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("Mean Annual Income and Spending Score for each Cluster:")
print(cluster_centers_df)

print("\n--- Visualizing the Clusters ---")

# Scatter plot of Annual Income vs Spending Score, colored by Cluster
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df,
                palette='viridis', s=100, alpha=0.8, legend='full')
# Plot the cluster centroids (optional, but very helpful)
# Get the centroids from the fitted KMeans model
# Centroids are in the scaled space, so we need to inverse transform them
# to the original scale for plotting on our original feature axes.
centroids_scaled = kmeans_model.cluster_centers_
centroids_original_scale = scaler.inverse_transform(centroids_scaled)
plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1],
            marker='X', s=500, color='red', label='Centroids', edgecolor='black')

plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.legend(title='Cluster')
plt.show() # Display the plot

print("\n--- Cluster Interpretation ---")
print("Based on the plot and cluster means, you can interpret the segments:")
print("  - For example, high income & high spending (likely 'target' customers)")
print("  - Low income & low spending (potential 'churn' risk or less engaged)")
print("  - Mid income & mid spending (average customers)")
# You will interpret these based on the actual plot results!

# Step 7: Save the Clustered Data and (Optional) the Model

# Save the DataFrame with cluster assignments to a new CSV file
output_filename = 'mall_customers_clustered.csv'
df.to_csv(output_filename, index=False)
print(f"\nClustered customer data saved to '{output_filename}'")

# Optional: Save the trained KMeans model
model_filename = 'kmeans_customer_segmentation_model.pkl'
joblib.dump(kmeans_model, model_filename)
print(f"Trained KMeans model saved as '{model_filename}'")

print("\n--- Project Complete ---")
print("You have successfully performed customer segmentation using K-Means Clustering!")
print("The 'mall_customers_clustered.csv' file now contains the customer data with their assigned cluster.")
print("The 'kmeans_customer_segmentation_model.pkl' file contains the trained clustering model.")