#################################################
#Unsupervised Learning : Unlabeled
    '''1)clustering: distance betewen points
            - plotting(visualizing)
            - 
        2)anomaly detections
        3)neural netoworks'''
        #ex) google news classfiy articles: using frequent terms in articles

#################################################
# Import plotting 
from matplotlib import pyplot as plt
plt.scatter(x, y)
plt.show()
#################################################       
#clustering: group of items with similar characteristics
    '''1)Customer segment based on their habbit
        2)Hierarchical clustering
        3) K-means clustering
        3) DBSCAN, Gaussian Methods'''
    #cluster center: mean of attributes of all data points
    #two closest cluester are merged, center of two merged cluesters are re-computed
    #repeated
#################################################

##################################################################################################       
#hierarchical clustering
    '''1) dataframe
        2)linkage: computes distances between intermediate clusters
        3)fcluster: generate clusters and assigned associated cluster labels to a new column
        4)plot'''
#K-means clustering
    '''1)random cluster center is generated for each of three clusters
        2)calculate distance from cluster center to each cluster points
        3)recomputed cluster centers (iteration with predefined # of times)'''
    #kmeans: computed centroids of the clusters
    #vq: cluster assignment for each points
##################################################################################################
#Pokémon sightings: hierarchical clustering
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
Z = linkage(df, 'ward')# Use the linkage() function to compute distance
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')# Generate cluster labels
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)# Plot 
plt.show()
#K-means
from scipy.cluster.vq import kmeans, vq
centroids,_ = kmeans(df,2)# Compute cluster centers
df['cluster_labels'], _ = vq(df, centroids)# Assign cluster labels
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)# Plot 
plt.show()
#################################################       
#Data preparation
    #variables have incomparable units
    #variables with different scales and variances
    #Data in raw form may lead to bias in clustering
    #Cluster may be heavily dependent on one variable
        #solution: normalization:rescaling data to a standard deviation of 1
                #=x/std_deb(x)
#################################################
#Normalization
from scipy.cluster.vq import whiten
goals_for = [4,3,2,3,1,1,2,0,1,4] 
scaled_data = whiten(goals_for) #standardize the data
print(scaled_data)

plt.plot(goals_for, label='original')# Plot original data
plt.plot(scaled_data, label='scaled')# Plot scaled data
plt.legend()# Show the legend in the plot
plt.show()

#FIFA 18: Normalize data
fifa['scaled_wage'] = whiten(fifa['eur_wage'])# Scale wage and value
fifa['scaled_value'] = whiten(fifa['eur_value'])
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')# Plot the two columns 
plt.show()
print(fifa[['scaled_wage', 'scaled_value']].describe())# Check mean and standard deviation
#################################################       
#Hierarchical clustering with scipy
    '''1)creating a distance matrix using linkage
        2) create cluster labels with ward method
    #method
            #single: uses the two closest objects between clusters to determine inter-cluster proximity --> more dispersed
            #complete:uses the two farthest objects among clusters to determine inter-cluster proximity
            #average:based on the arithmetic mean of all objects
            #centroid: based on the geometric mean of all objects
            #median: based on the median 
            #ward: based on the sum of squares --> ususally dense toward the center
    *Euclidean distance: straight line distance between two points on a 2D plane'''
#################################################  
#ward method
from scipy.cluster.hierarchy import fcluster, linkage
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean')# Use the linkage() function
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')# Assign cluster labels
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
#################################################       
#Visualiza cluster?
    #to validate the cluster
    #spot trends in data
#################################################               
#matplotlib
from matplotlib import pyplot as plt
colors = {1:'red', 2:'blue'}# Define a colors dictionary for clusters
comic_con.plot.scatter(x = 'x_scaled', 
                	   y = 'y_scaled',
                	   c = comic_con['cluster_labels'].apply(lambda x:colors[x]))
plt.show()

#sesaborn
import seaborn as sns
sns.scatterplot(x = 'x_scaled', 
                	   y = 'y_scaled', 
                hue='cluster_labels', 
                data = comic_con)
plt.show()
#################################################       
#How to decide how many cluster?
    #Dendrogram
        #-each step: merging of two cloest clusters in earlier step
        #-X axis: individual points, y: distance between clusters
        #-width of U: distance between two child clusters
            #-horizontal line: # of vertical lines it intersets = # of clusters at that stage 
            #-distance between horizontal line = inter-cluster distance
################################################# 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster, linkage
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean')# Use the linkage() function
dn = dendrogram(distance_matrix)# Create a dendrogram
plt.show()
#################################################       
#Limitation
    #huge data not appropriate for Hierarchical clustering: linkage takes too long
################################################# 
# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', hue='cluster_labels', data=fifa)
plt.show()
##################################################################################################   
#K-clustering
    ''' 1)generate cluster centers: kmeans (single value of distortions)
          distortion = sum of square of distances from each cluster centers
        2)generate cluster labels: vq(standardized observation, cluster centers, check_finite) = a list of distortions'''
##################################################################################################       
from scipy.cluster.vq import kmeans, vq
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']],2)# Generate cluster centers
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']],cluster_centers)# Assign cluster labels
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()
%timeit kmeans(comic_con[['x_scaled', 'y_scaled']],2) #run time
#################################################       
#How many cluster?
    #No absolute method to find # of cluster
    #1)Elblow method: plot of # of clusters and distortion
        #Distortions revisited: distortion: sum of squared distances of points from cluster centers
            #decreases with an increasing # of clusters
            #becomes zero when # of cluester = # points
    #2)Average silhouette
    #3)Gap statistic
################################################# 
#Elbow method on distinct clusters
distortions = []
num_clusters = range(1, 7)

for i in num_clusters:
    cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']],i)
    distortions.append(distortion) #create a list of distortion

elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()
#Elbow method on uniform data
for i in num_clusters:
    cluster_centers, distortion = kmeans(uniform_data[['x_scaled', 'y_scaled']],i)
    distortions.append(distortion)
#################################################       
#Limitation
#################################################