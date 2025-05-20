# Customer Segmentation using K-Means Clustering (Mall Customers)

## Project Overview

This project demonstrates an unsupervised machine learning approach to segment retail customers based on their purchasing behavior. By applying the K-Means clustering algorithm, we identify distinct groups of customers, enabling targeted marketing strategies and a deeper understanding of customer demographics and spending habits.

The project uses the `Mall_Customers.csv` dataset, focusing on two key features: **Annual Income** and **Spending Score**.

## Features Used

* **`Annual Income (k$)`**: The customer's annual income in thousands of dollars.
* **`Spending Score (1-100)`**: A score assigned by the mall based on customer behavior and spending habits, with 100 being the highest spending.
* **Target (for clustering)**: No explicit target variable as this is unsupervised learning. The goal is to find inherent groups within the data.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.x installed on your system.

### 1. Project Setup

* Create a dedicated folder for this project, e.g., `Customer Segmentation using K-Means Clustering`.
* Place the main Python script (`customer_segmentation.py`) and the dataset (`Mall_Customers.csv`) inside this folder.

### 2. Download the Dataset

The project uses the `Mall_Customers.csv` dataset.

* You can download it directly from Kaggle: [Mall Customer Segmentation Data](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
* Ensure the downloaded `Mall_Customers.csv` file is placed directly into your project folder.

### 3. Install Dependencies

It's highly recommended to use a virtual environment to manage project dependencies.

* **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    ```
* **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
* **Install required packages:**
    The `requirements.txt` file lists all necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the necessary libraries, including `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.

## How to Run the Project

The `customer_segmentation.py` script performs all steps from data loading and preprocessing to model training, evaluation, visualization, and saving the results.

* Open your terminal or command prompt.
* Navigate to your project directory:
    ```bash
    cd "Customer Segmentation using K-Means Clustering"
    ```
* Run the main script:
    ```bash
    python customer_segmentation.py
    ```

    This script will:
    * Load the `Mall_Customers.csv` dataset.
    * Perform Exploratory Data Analysis (EDA) and display histograms and a scatter plot of the key features.
    * Scale the features (`Annual Income (k$)` and `Spending Score (1-100)`).
    * Apply the **Elbow Method** to help determine the optimal number of clusters (`K`), displaying a plot for your analysis. (For this dataset, `K=5` is usually the optimal choice).
    * Train the K-Means model with the chosen `K`.
    * Analyze the characteristics of each cluster by printing their mean income and spending score.
    * Visualize the clusters on a scatter plot, showing distinct customer segments.
    * Save the original dataset with a new `Cluster` column to `mall_customers_clustered.csv`.
    * Save the trained K-Means model to `kmeans_customer_segmentation_model.pkl`.

## How Retailers Can Use This Project

This project's primary output for a retailer is the **`mall_customers_clustered.csv` file** and the **insights derived from the cluster analysis**.

1.  **Understanding Customer Segments:** The retailer (e.g., marketing manager) would review the cluster characteristics (mean income and spending score for each group) and the visualization. They would then give meaningful names to each segment (e.g., "High-Value Shoppers," "Budget-Conscious Spenders," "Potential VIPs").
2.  **Targeted Marketing Campaigns:** Using the `mall_customers_clustered.csv` file, the retailer can link each `CustomerID` to their assigned `Cluster`. They can then filter their customer database and create specific marketing campaigns tailored to each segment's behavior:
    * **High-Value Segments:** Offer exclusive discounts, loyalty rewards, or early access to new products.
    * **Low-Spending Segments:** Send promotional offers, introduce value-for-money items, or win-back campaigns.
    * **Specific Income/Spending Segments:** Tailor product recommendations or communication style.
3.  **Personalized Customer Experience:** The cluster information can inform sales associates or customer service representatives about a customer's likely preferences and value, allowing for more personalized interactions.
4.  **Strategic Decision Making:** Insights from customer segmentation can influence product development, inventory management, store layout, and overall business strategy.

The retailer does *not* typically input individual customer data into the `customer_segmentation.py` script daily. Instead, the script is run periodically (e.g., monthly or quarterly) on the entire customer database to re-segment and update insights as customer behavior evolves.

