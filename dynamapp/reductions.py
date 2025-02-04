import pandas as pd
import jax.numpy as jnp
import jax

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    
    PCA projects the data into a lower-dimensional space while retaining as much variance as possible.
    """
    def __init__(self, n_components: int):
        """
        Args:
            n_components (int): Number of principal components to keep after the transformation.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X: jnp.ndarray):
        """
        Fit the PCA model to the data.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
        """
        self.mean = jnp.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = jnp.cov(X_centered, rowvar=False)
        
        eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)
        
        sorted_indices = jnp.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[sorted_indices]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        self.components = eigvecs_sorted[:, :self.n_components]
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Apply dimensionality reduction to the data using the fitted PCA model.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: The transformed data of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return jnp.dot(X_centered, self.components)
    
    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the PCA model and apply dimensionality reduction in one step.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: The transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

class LDA:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction.
    
    LDA is supervised and finds a linear combination of features that 
    best separates the classes.
    """
    def __init__(self, n_components: int):
        """
        Args:
            n_components (int): Number of discriminants to keep after the transformation.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.class_means = None
        self.class_cov = None
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        """
        Fit the LDA model to the data.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
            y (jnp.ndarray): The labels for the data of shape (n_samples,).
        """
        classes = jnp.unique(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        self.class_means = jnp.array([jnp.mean(X[y == c], axis=0) for c in classes])
        
        Sw = jnp.zeros((n_features, n_features))
        for c in classes:
            X_c = X[y == c]
            mean_c = self.class_means[c]
            Sw += jnp.dot((X_c - mean_c).T, (X_c - mean_c))
        
        overall_mean = jnp.mean(X, axis=0)
        Sb = jnp.zeros((n_features, n_features))
        for c in classes:
            n_c = X[y == c].shape[0]
            mean_c = self.class_means[c]
            Sb += n_c * jnp.outer(mean_c - overall_mean, mean_c - overall_mean)
        
        eigvals, eigvecs = jnp.linalg.eigh(jnp.linalg.inv(Sw).dot(Sb))
        
        sorted_indices = jnp.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[sorted_indices]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        
        self.components = eigvecs_sorted[:, :self.n_components]
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Apply dimensionality reduction to the data using the fitted LDA model.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: The transformed data of shape (n_samples, n_components).
        """
        return jnp.dot(X - jnp.mean(X, axis=0), self.components)
    
    def fit_transform(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the LDA model and apply dimensionality reduction in one step.
        
        Args:
            X (jnp.ndarray): The input data of shape (n_samples, n_features).
            y (jnp.ndarray): The labels for the data of shape (n_samples,).
        
        Returns:
            jnp.ndarray: The transformed data of shape (n_samples, n_components).
        """
        self.fit(X, y)
        return self.transform(X)