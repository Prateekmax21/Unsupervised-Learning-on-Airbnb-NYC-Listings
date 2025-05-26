# Clustering the City: Uncovering Patterns in NYC Airbnb Listings with Unsupervised Learning

## 📌 Introduction

Airbnb has completely transformed travel by making it easier for people to rent out their homes or rooms to guests looking for short-term stays. This creates a huge, messy, but super interesting pool of data. In this project, we asked: **Are there hidden patterns among Airbnb listings in NYC?** Could we find categories like "budget places that are always available" or "expensive ones run like hotels"?

To dig into this, we used real NYC Airbnb data and applied **unsupervised learning**—that means we didn't tell the model what to look for, we let it discover the structure on its own. We focused on features like price, availability, number of reviews, and how many listings a host has.

## 🧠 Theoretical Background

### Principal Component Analysis (PCA)
Think of PCA like condensing all the information from 10 columns into just 2 or 3, while keeping most of what matters. It makes visualizations easier and helps remove noise.

### Singular Value Decomposition (SVD)
This is the mathematical engine behind PCA. It helps break down complex datasets and is great for filling in missing data, like if a review count is blank.

### K-Means Clustering
This groups listings into clusters based on their similarity. We choose a number `k` (e.g., 3 clusters), and it assigns each listing to one of the clusters by trying to minimize the distance within groups.

### Hierarchical Clustering
Instead of pre-choosing `k`, this builds a tree (dendrogram) showing how all listings relate. We can “cut” this tree at different heights to form groups.

### Matrix Completion
Real-world data has gaps. To simulate this, we randomly removed 5% of values and used PCA to predict/fill them. This shows how robust PCA can be even when data isn’t perfect.

## 💜 Methodology

### Data Cleaning
We picked features we care about:
- Price
- Minimum & Maximum nights
- Number of reviews & reviews per month
- Availability in a year
- Number of listings a host manages
- Accommodates, bathrooms, and bedrooms

Then we dropped rows with missing or blank values, filtered extreme outliers (like $10,000 listings), and scaled the data so each feature is equally weighted using `StandardScaler`.

### PCA + Variance Explained
We used PCA to reduce dimensions and found that:
- PC1 explains ~26% of variance
- First 3 components explain ~58%
- All 10 together explain 100%

We used line plots to show this instead of bars—cleaner and easier to read.

### Matrix Completion
We simulated missing values, ran PCA with 5 components, and then filled them in. The correlation between the actual and predicted values was ~0.51—not bad!

### Clustering Steps
- We projected the data into PC1 and PC2 for clustering.
- K-Means with `k=3` gave distinct, meaningful clusters.
- Hierarchical clustering was also applied using complete, average, and single linkage methods.

## 📊 Results and Visualizations
- Scree Plot 
![PCA](https://github.com/user-attachments/assets/7eb64427-7dce-4657-9579-f83248792c06)
- Single linkage Dendogram
![single linkage](https://github.com/user-attachments/assets/656815c4-3236-42c5-8ea7-a4d29220d3bb)
- Average linkage Dendogram
![average linkage](https://github.com/user-attachments/assets/9479783f-050d-45b9-a99e-41f752eefd1f)
- Complete linkage Dendogram
![complete linkage](https://github.com/user-attachments/assets/8e160b3d-41f0-4b55-b5cf-fcdaf238fa64)
- Complete linkage Dendogram with cut height 4.5
![complete linkage with cut height 4 5](https://github.com/user-attachments/assets/a8eb4db3-1ebb-493c-bced-555211c257d6)
- Elbow & Silhouette plots
![Kmeans clusters](https://github.com/user-attachments/assets/72f5d65e-3b1f-49c5-bc96-0e962ba8cda0)
- K-Means cluster scatter
![cluster k=3](https://github.com/user-attachments/assets/0b5935c8-8af5-4eab-93bd-0a27cae46689)


## 💭 Discussion

We got a lot out of this unsupervised dive:
- PCA helped reduce noise and highlight what's important
- The first few PCs captured most of the meaningful variation
- K-Means found three clear types of listings:
  1. Budget listings with high availability
  2. Expensive listings likely run by professional hosts
  3. Low-activity listings for longer stays

Hierarchical clustering validated our groups and helped us explore alternatives.

This info is gold for Airbnb hosts (how to price and position), travelers (what to expect), and even city planners (how short-term rentals impact neighborhoods).

## 📅 Conclusion

Even without knowing anything about the listings ahead of time (no labels!), we uncovered natural groupings in NYC's Airbnb data. These methods—especially PCA and clustering—gave us structure, insights, and visualizations from messy real-world data.

Unsupervised learning is powerful because it finds stories we didn’t even know to look for.

## 📒 References

1. Inside Airbnb Dataset: [http://insideairbnb.com/get-the-data.html](http://insideairbnb.com/get-the-data.html)  
2. scikit-learn documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
3. Tan, P.-N., Steinbach, M., & Kumar, V. (2019). *Introduction to Data Mining*  
4. Jolliffe, I. T. (2002). *Principal Component Analysis*, Springer  
"""
