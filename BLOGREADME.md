# Clustering the City: Uncovering Patterns in NYC Airbnb Listings with Unsupervised Learning

## Introduction

Airbnb has completely changed how people travel. Instead of booking hotels, travelers can now rent apartments, spare rooms, or even couches directly from locals. But have you ever wondered: Are there patterns in these listings? Do budget-friendly rentals behave differently than luxury ones? Are there different "types" of hosts?

That's what we set out to discover in this project. We used real Airbnb data from New York City and applied unsupervised learning techniques—methods that help us explore and group data without knowing the answers in advance. Our goal was to find natural clusters of listings and figure out what makes them different.

We looked at things like price, number of reviews, how often a place is available, and how many properties a host manages. Using tools like PCA, K-Means, and Hierarchical Clustering, we tried to uncover what types of listings exist in NYC's Airbnb market.

## Theoretical Background

### Principal Component Analysis (PCA)

PCA is like smart compression. It reduces the number of features in our data while keeping most of the important information. It gives us new axes (principal components) that explain how our data varies. We use these components to simplify analysis and make cool 2D visualizations.

### Singular Value Decomposition (SVD)

SVD is the math behind PCA. It breaks a matrix into three parts: U, Σ, and V*. U helps us understand each row (i.e., each listing), and V* tells us how features contribute to patterns. PCA and SVD give similar results when data is normalized.

### K-Means Clustering

This technique groups listings based on similarity. It randomly assigns centers and iteratively refines them so each listing belongs to the nearest center. We tested different values of k (number of clusters) and chose the best using silhouette score.

### Hierarchical Clustering

This method builds a tree of clusters (called a dendrogram). We don't have to choose the number of clusters in advance. We just "cut" the tree at a certain height to get groups. We used different ways (linkages) to calculate how to merge clusters: complete, average, and single.

### Matrix Completion

Since real-world data is messy, we simulated missing values and used PCA to fill them in. This showed us how PCA can still work when the dataset isn’t perfect.

## Methodology

### Cleaning

We focused on key columns:
- price
- minimum_nights
- maximum_nights
- number_of_reviews
- reviews_per_month
- availability_365
- calculated_host_listings_count
- accommodates
- bathrooms
- bedrooms

We dropped listings with missing or blank values and filtered out outliers (e.g., super high price or 1000-night minimum stays). We standardized the data using StandardScaler so every feature has the same importance.

### PCA and Variance Analysis

PCA showed that:
- The first component explains 26% of the variance
- First 3 components explain ~58%
- All 10 components explain 100% (obviously, because we started with 10 features)

We used line plots (not bar graphs) for cleaner visualization of variance explained.

### Matrix Completion

We randomly removed 5% of values, applied PCA with 5 components, and reconstructed the missing values. We got a correlation of ~0.51 between the original and filled-in values. That’s pretty good for an imputation!

### Clustering

We projected the data onto PC1 and PC2 and applied:

#### K-Means

- k=3 gave good silhouette scores
- Clusters were distinct when plotted in 2D

#### Cluster Interpretation

- Cluster 0: Affordable, small listings
- Cluster 1: Large, expensive listings (commercial hosts)
- Cluster 2: Long-stay listings, fewer reviews

#### Hierarchical Clustering

We used complete, average, and single linkage to build dendrograms. Cutting the tree at 3 clusters gave similar groups to K-Means. Complete linkage gave the clearest separation.

## Results and Visualizations

Include these plots in your notebook:
- Scree Plot (PVE vs PC)
- Cumulative PVE
- Scatter of PC1 vs PC2
- K-Means cluster scatter
- Cluster centroids table
- Dendrograms (Complete, Average, Single)

## Discussion

- PCA helped reduce dimensionality and noise
- The loadings (V*) told us which features matter most (price, availability, reviews)
- K-Means gave clean, interpretable clusters
- Hierarchical clustering revealed nested groupings

Comparing the clusters helped us understand different behaviors:
- High-end listings managed by pros
- Budget-friendly, frequently reviewed properties
- Listings with long stays but low activity

These insights can help Airbnb, hosts, and even city planners make smarter decisions.

## Conclusion

Unsupervised learning revealed clear patterns in NYC Airbnb data. Without any labels, we discovered 3 main types of listings, learned which features separate them, and gained insights into different rental strategies.

This approach can be scaled to other cities or combined with geolocation data for even more insights. It’s a great example of how data science can turn raw data into useful information.

## References

- Inside Airbnb Dataset: http://insideairbnb.com/get-the-data.html
- scikit-learn documentation: https://scikit-learn.org/stable/
- Clustering in ML : https://scikit-learn.org/stable/modules/clustering.html
- ISLP Textbook and Lecture Code: Ch12-1.ipynb, Ch12-2.ipynb
- PCA: https://www.geeksforgeeks.org/principal-component-analysis-pca/
- Tan, P.-N., Steinbach, M., & Kumar, V. (2019). Introduction to Data Mining
- Jolliffe, I. T. (2002). Principal Component Analysis. Springer
"""
