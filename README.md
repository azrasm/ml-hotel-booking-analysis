# Hotel booking demand - Data analysis, classification and clustering
This project provides a comprehensive Data Science pipeline for analyzing hotel booking data and predicting cancellations. It covers the full lifecycle from exploratory data analysis (EDA) and data cleaning to building and optimizing a **k-Nearest Neighbors (k-NN)** classification model.

## Dataset
The dataset consists of 119,390 rows and 30 columns, featuring:
* 12 Categorical Variables (meal type, distribution channel, customer type, etc.).
* 18 Numerical Variables (lead time, ADR, number of stays, etc.)

Target Variable: _is_canceled_ (Binary: 0 = Not Canceled, 1 = Canceled).

## Exploratory Data Analysis
* Correlation Analysis: Performed using Pearson’s Correlation Coefficient to identify relationships between numerical features.
* Distribution Visualization: Histograms with KDE were used to analyze data density and distribution shifts after cleaning.
* Feature Engineering & Selection:
    * Removed _reservation_status_ and _reservation_status_date_ to prevent data leakage (as they directly reveal the outcome).
    * Dropped the _country_ variable to reduce high dimensionality, which significantly improves k-NN performance by mitigating the "curse of dimensionality."
 
## Modeling Pipeline
The core of this project is a robust Scikit-Learn Pipeline that ensures clean data flow and prevents data leakage during cross-validation.
1. Preprocessing
  *  Scaling: Numerical features were standardized using StandardScaler.
  *  Encoding: Categorical variables were transformed using OneHotEncoder.
  *  Data Splitting: 70% Train / 15% Validation / 15% Test, using Stratified Sampling to maintain class proportions.

2. Handling Class Imbalance

The dataset is imbalanced (more non-cancellations than cancellations). Two techniques were implemented:
 * SMOTE (Oversampling): Generating synthetic examples for the minority class.
 * RandomUnderSampler (Undersampling): Balancing the classes by reducing majority class samples.
 
## Hyperparameter Optimization

To maximize model performance, HalvingGridSearchCV is utilized. This iterative elimination approach is more efficient than standard GridSearch for large datasets.
Optimized Parameters:
* Distance Metric: cityblock (Manhattan)
* Weights: uniform
* Neighbors (k): 9

Resulting Accuracy: 71.44% (Optimized for F1-Score).

## Evaluation Metrics

The model was evaluated using a multi-metric approach to ensure reliability:
+ Confusion Matrix: Detailed breakdown of TP, TN, FP, FN.
+ Precision & Recall: Balancing the cost of false positives vs. the importance of catching all cancellations.
+ F1-Score & Cohen’s Kappa: Measuring performance beyond simple accuracy.
+ ROC-AUC: Assessing the model’s ability to distinguish between classes.
+ Cross-Validation (k=5): Used to assess stability and generalization, reducing the risk of overfitting.

  # Customer Segmentation
  
After the predictive modeling, Hierarchical Clustering is performed to discover natural patterns and behavioral segments within the guest data.
1. Clustering Methodology
   * Feature Selection: Focused on numerical variables (Lead Time, ADR, stays) to identify planning and spending patterns while reducing noise from high-dimensional categorical data.
   * Data Preparation:
       * Removed _is_canceled _and status-related columns to ensure true unsupervised learning.
       * Filtered anomalies (e.g., bookings with 0 guests or ADR > 1000) to protect the Ward Linkage method from outliers.
       * Applied StandardScaler to normalize features.
   * Optimization: Due to high computational complexity, a representative 30% random sample of the dataset was used.
   * Linkage Method: Utilized Ward’s Method to minimize intra-cluster variance.
     
2. Identifying Clusters

The optimal number of clusters was determined using a Dendrogram. By identifying the largest vertical distance between merges, 3 distinct clusters were selected.

**Evaluation:** Achieved a Silhouette Score of 0.279. This is a robust result for real-world behavioral data where boundaries are naturally fluid.

## Cluster Profiles & Business Insights

The analysis identified three stable, business-relevant segments:

| Feature | Cluster 0: Loyal Returnees | Cluster 1: Premium Families | Cluster 2: Strategic Planners |
| :--- | :--- | :--- | :--- |
| **Profile Summary** | Highly loyal, price-sensitive, and last-minute bookers. | High-spending families with specific amenity requirements. | Reliable, long-term planners providing occupancy stability. |
| **Key Metric (Z-Score)** | **is_repeated_guest (5.2)** | **children (1.8)** & **babies (0.57)** | **lead_time (Positive deviation)** |
| **Average ADR** | Lowest (€64.99) | Highest (€134.69) | Average (~€100) |
| **Lead Time** | Shortest (-0.61 std) | Moderate | Longest (110.18 days) |
| **Special Requests** | Minimal | Highest (0.87 avg) | Average/Stable |
| **Parking Needs** | Low | High (1.7 std) | Low |
| **Marketing Strategy** | Flash sales & loyalty discounts. | Premium packages & family-oriented services. | Early-bird promotions & seasonal planning. |

## Dimensionality Reduction & Visualization

To verify the stability of these segments, I applied Principal Component Analysis (PCA) and generated a Z-score Heatmap.

* PCA Visualization: Confirmed three distinct "data clouds." Cluster 1 (Premium) showed high homogeneity (dense grouping), while Cluster 2 (Strategic) showed wider variance in planning habits.
* Heatmap Analysis: Used Z-score scaling (Blue-White-Red) to visualize deviations from the mean.
   * Cluster 0 showed an intense positive deviation for is_repeated_guest.
   * Cluster 1 spiked in children, babies, and required_car_parking_spaces.
   * Cluster 2 remained neutral, representing the "standard" guest behavior.

## Conclusion & Business Impact

This research successfully applied Hierarchical Clustering to a representative sample of hotel booking data. While the Silhouette Coefficient of 0.279 reflects the inherent complexity and overlap of real-world behavioral data, the resulting clusters demonstrated high stability and clear interpretative value.

Key Takeaways:
* Methodological Integrity: Using the original scaled dataset (rather than PCA-transformed features) for final profiling proved to be the correct decision, as it allowed for a direct clinical interpretation of guest attributes.
* Hypothesis Validation: All initial hypotheses regarding guest segmentation based on lead time, spending (ADR), and loyalty were confirmed.
* Strategic Recommendations:
    + For Cluster 0 (Loyalists): Implement "last-minute" flash sales and loyalty-based discounts due to their short booking cycles.
    + For Cluster 1 (Premium): Focus on resource allocation (parking, childcare) and hyper-personalization, as this segment yields the highest daily revenue.
    + For Cluster 2 (Planners): Utilize early-bird promotions to secure baseline occupancy well in advance.

In summary, Hierarchical Clustering acted as a powerful diagnostic tool, transforming raw statistical data into actionable guest personas that directly impact hotel profitability and operational efficiency.

## References
**dataset:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?fbclid=IwY2xjawN4JXxleHRuA2FlbQIxMABicmlkETBQclljV1lCSGd0T0xkcWlCc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHhqapL6apOukKY8d32xG1WKqSjp7E12CRyNoma8mowc29B5xy21qbKBlcriB_aem_D1j755PhdE1DPyJxotd9Ag
