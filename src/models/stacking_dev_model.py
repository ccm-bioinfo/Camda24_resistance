import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler


from utils import RoundLog2Transformer

class ClusterAugmentEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3):
        # Initialize the Gaussian Mixture Model and Agglomerative Clustering model
        self.n_clusters = n_clusters
        self.clustering_gmm = GaussianMixture(n_components=n_clusters)
        self.clustering_agg = AgglomerativeClustering(n_clusters=n_clusters)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        
    def fit(self, X, y=None):
        """
        Fit the GMM and Agglomerative clustering models to the data.
        GMM generates cluster probabilities and Agglomerative Clustering labels the data.
        """
        # Fit the GMM model to the data
        self.clustering_gmm.fit(X)
        
        # Fit the Agglomerative Clustering model to the data
        self.clustering_agg.fit(X)
        
        # Get the Agglomerative clustering labels
        agg_labels = self.clustering_agg.labels_
        
        # Set the centroids of Agglomerative clusters
        self.clustering_agg.centroids = np.array([
            X[agg_labels == i].mean(axis=0) for i in range(self.n_clusters)
        ])
        
        # One-hot encode the Agglomerative labels
        self.one_hot_encoder.fit(agg_labels.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        """
        Transform the input data into a feature set with GMM probabilities and one-hot encoded Agglomerative labels.
        """
        # Predict cluster probabilities using GMM for the test data
        gmm_probs = self.clustering_gmm.predict_proba(X)
        
        # Get the predicted Agglomerative labels by finding the closest centroid
        agg_labels, _ = pairwise_distances_argmin_min(X, self.clustering_agg.centroids)
        
        # One-hot encode the Agglomerative labels
        agg_labels_onehot = self.one_hot_encoder.transform(agg_labels.reshape(-1, 1))
        
        # Combine original features with the clustering features
        X_with_clusters = np.column_stack([X, gmm_probs, agg_labels_onehot])
        
        return X_with_clusters
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and then transform the data into the desired feature set.
        """
        self.fit(X, y)
        return self.transform(X)


class MultiOutputModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self,
                 base_estimators_reg=None,
                 base_estimators_clf=None,
                 meta_regressor=None,
                 meta_classifier=None,
                 num_clusters=3,
                 use_oversampling=True,
                 use_stacking=True,
                 n_splits=5
        ):
        """
        Initialize the multi-output model with optional base models and hyperparameters.
        
        Parameters:
        - base_estimators_reg: List of base regressors.
        - base_estimators_clf: List of base classifiers.
        - meta_regressor: Meta model for regression.
        - meta_classifier: Meta model for classification.
        - num_clusters: Number of clusters for clustering models.
        - use_oversampling: Whether to apply oversampling for classification.
        - use_stacking: Whether to use stacking for regression and classification.
        """
        self.use_oversampling = use_oversampling
        self.use_stacking = use_stacking
        self.num_clusters = num_clusters
        self.n_splits = n_splits
        # Default base estimators if not provided
        self.base_estimators_reg = base_estimators_reg 
        
        self.base_estimators_clf = base_estimators_clf
        
        # Default meta models if not provided
        self.meta_regressor = meta_regressor 
        self.meta_classifier = meta_classifier        

        # Clustering models
        self.augmenter_with_clustering = ClusterAugmentEncoder(n_clusters=self.num_clusters) 

        # Function to transform regression predictions
        self.log2_transformer = RoundLog2Transformer()

        # StratifiedKFold for stacking classifier
        self.cv_clf = StratifiedKFold(n_splits=self.n_splits)         

    @staticmethod
    def _oversample_data(X, y_reg, y_clf):
        """Oversample classification targets while maintaining regression targets' relationship."""
        ros = RandomOverSampler()
        X_res, y_res_clf = ros.fit_resample(X, y_clf)
        
        # For regression targets, we match them using oversampled indices
        idx_resampled = ros.sample_indices_  
        y_res_reg = y_reg[idx_resampled]
        
        return X_res, y_res_clf, y_res_reg


    def fit(self, X, y):
        # Separate target into regression and classification tasks
        y_reg = y[:, 0]
        y_clf = y[:, 1:3]

        # Optional Oversampling
        if self.use_oversampling:
            X, y_clf, y_reg = MultiOutputModel._oversample_data(X, y_reg, y_clf)

        # Fit the clustering models
        X_with_clusters = self.augmenter_with_clustering.fit_transform(X)

        # Conditionally use stacking or just random forests
        if self.use_stacking:
            # Create Stacking Models
            self.stack_reg = StackingRegressor(estimators=self.base_estimators_reg, final_estimator=self.meta_regressor)
            self.stack_clf = StackingClassifier(estimators=self.base_estimators_clf, final_estimator=self.meta_classifier, cv=self.cv_clf)
            
            # Fit the models with the new feature set
            self.stack_reg.fit(X_with_clusters, y_reg)
            self.stack_clf.fit(X_with_clusters, y_clf)
        else:
            # Use only RandomForest if stacking is disabled
            self.rf_reg = RandomForestRegressor(class_weight='balanced')
            self.rf_clf = RandomForestClassifier(class_weight='balanced')
            
            # Fit the Random Forest models
            self.rf_reg.fit(X_with_clusters, y_reg)
            self.rf_clf.fit(X_with_clusters, y_clf)
        return self

    def predict(self, X):
        # Transform the input data with clustering models
        X_with_clusters = self.augmenter_with_clustering.transform(X)

        # Predict with the models
        if self.use_stacking:
            reg_pred = self.stack_reg.predict(X_with_clusters)
            clf_pred_prob = self.stack_clf.predict_proba(X_with_clusters)[:, 1]
        else:
            reg_pred = self.rf_reg.predict(X_with_clusters)
            clf_pred_prob = self.rf_clf.predict_proba(X_with_clusters)[:, 1]

        # Apply inverse log2 transformation to regression predictions
        reg_pred_transformed = self.log2_transformer.inverse_transform(reg_pred.reshape(-1, 1)).flatten()
        return np.column_stack([reg_pred_transformed, clf_pred_prob])


if __name__ == "__main__":
    # Example data
    X_example = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ])
    y_example = np.array([
        [10, 0, 1],
        [12, 0, 1],
        [15, 1, 0],
        [20, 1, 0],
        [5, 0, 1],
        [25, 1, 0]
    ])

    # Instantiate the MultiOutputModel
    model = MultiOutputModel(
        base_estimators_reg=[('rf', RandomForestRegressor())],
        base_estimators_clf=[('rf', RandomForestClassifier())],
        meta_regressor=RandomForestRegressor(),
        meta_classifier=RandomForestClassifier(),
        num_clusters=2,
        use_oversampling=False,
        use_stacking=True,
        n_splits=3
    )

    # Fit the model
    model.fit(X_example, y_example)

    # Predict using the model
    predictions = model.predict(X_example)
    print(predictions)