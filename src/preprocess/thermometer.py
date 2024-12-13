import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ThermometerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, self.ascending_categories_=None):
        """
        Initialize the ThermometerTransformer with no hyperparameters.
        """
        self.ascending_categories_ = self.ascending_categories_

    def fit(self, data: np.ndarray):
        """
        Fit the thermometer transformer by determining the unique categories in the data.

        Parameters:
        data (np.ndarray): Array of ordinal data to fit the transformer.

        Returns:
        self: The fitted transformer.
        """
        #unique categories (levels) in the data, sorted ascending ordinally
        if self.ascending_categories_ is None:
            self.ascending_categories_ = np.unique(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the ordinal data to thermometer encoding (binary vectors).

        Parameters:
        data (np.ndarray): Array of ordinal data to transform.

        Returns:
        np.ndarray: The transformed binary matrix (thermometer encoding).
        """
        if self.ascending_categories_ is None:
            raise ValueError("ThermometerTransformer not fitted. Call fit() before transform.")
        
        # Initialize the binary matrix
        n_samples = data.shape[0]
        n_categories = len(self.ascending_categories_)
        encoded_data = np.zeros((n_samples, n_categories), dtype=int)

        # Fill the binary matrix with 1's where the category matches
        for i, value in enumerate(data):
            idx = np.searchsorted(self.ascending_categories_, value)  # Find the index of the value
            encoded_data[i, :idx + 1] = 1  # Set the corresponding positions to 1

        return encoded_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the thermometer encoding back to ordinal data.

        Parameters:
        data (np.ndarray): Array of binary vectors to inverse transform.

        Returns:
        np.ndarray: The inverse transformed ordinal data.
        """
        # Check that fit() has been called before inverse_transform()
        if self.ascending_categories_ is None:
            raise ValueError("ThermometerTransformer not fitted. Call fit() before inverse_transform.")
        
        # Determine the ordinal data from the binary matrix
        ordinal_data = np.array([self.ascending_categories_[np.argmax(row)] for row in data])
        return ordinal_data

# Example usage outside the class
if __name__ == "__main__":
    # Example Data
    data = np.array([1, 2, 3, 2, 1, 3, 1, 2, 3])

    # Instantiate the ThermometerTransformer
    transformer = ThermometerTransformer()

    # Fit and transform the data
    transformer.fit(data)
    encoded_data = transformer.transform(data)

    print("Original Data:")
    print(data)
    print("\nUnique Categories:")
    print(transformer.categories_)
    print("\nThermometer Encoded Data:")
    print(encoded_data)