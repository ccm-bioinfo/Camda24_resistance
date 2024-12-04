from imblearn.over_sampling import RandomOverSampler

class Oversampler:
    def __init__(self, x_transformed, y_transformed):
        self.x_transformed = x_transformed
        self.y_transformed = y_transformed

    def random_oversample(self, target_idx=0):
        if target_idx not in [0, 1]:
            raise ValueError("target_idx must be 0 or 1.")

        y_target = self.y_transformed[:, target_idx]
        ros = RandomOverSampler()
        x_resampled, _ = ros.fit_resample(self.x_transformed, y_target)
        indices = ros.sample_indices_
        y_resampled = self.y_transformed[indices]

        return x_resampled, y_resampled

if __name__ == "__main__":
    import numpy as np
    x_transformed = np.random.rand(10, 5)
    y_transformed = np.array([
        [0, 1], [1, 0], [0, 0], [1, 1],
        [0, 1], [1, 0], [0, 1], [1, 0],
        [0, 0], [1, 1]
    ])

    oversampler = Oversampler(x_transformed, y_transformed)
    x_resampled, y_resampled = oversampler.random_oversample(target_idx=0)

    print("Original X shape:", x_transformed.shape)
    print("Original y_transformed shape:", y_transformed.shape)
    print("Resampled X shape:", x_resampled.shape)
    print("Resampled y_transformed shape:", y_resampled.shape)