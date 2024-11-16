import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

class BagOfVisualSounds(BaseEstimator, TransformerMixin):
    def __init__(self, k=100):
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.centroids = None
        
    