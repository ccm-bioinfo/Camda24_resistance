import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class TwoStageVAE(Model):
    def __init__(self, input_dim, latent_dim, hidden_units=64):
        super(TwoStageVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(latent_dim * 2)  # Mean and log-variance for latent space
        ])

        # Decoders
        # Stage 1 decoders (dominant prediction)
        self.regressor_stage1 = self._build_decoder(hidden_units, 1, activation='linear')
        self.classifier_stage1 = self._build_decoder(hidden_units, 1, activation='sigmoid')

        # Stage 2 decoders (refinement for non-dominant samples)
        self.regressor_stage2 = self._build_decoder(hidden_units, 1, activation='linear')
        self.classifier_stage2 = self._build_decoder(hidden_units, 1, activation='sigmoid')

    def _build_decoder(self, hidden_units, output_dim, activation):
        """Builds a decoder network."""
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.latent_dim,)),
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(output_dim, activation=activation)
        ])

    def encode(self, x):
        """Encodes input into latent space."""
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Applies the reparameterization trick."""
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, stage):
        """Decodes latent representation."""
        if stage == 1:
            reg_pred = self.regressor_stage1(z)
            clf_pred = self.classifier_stage1(z)
        elif stage == 2:
            reg_pred = self.regressor_stage2(z)
            clf_pred = self.classifier_stage2(z)
        else:
            raise ValueError("Stage must be 1 or 2.")
        return reg_pred, clf_pred

    def call(self, x, stage):
        """Full VAE pipeline."""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reg_pred, clf_pred = self.decode(z, stage)
        return reg_pred, clf_pred, mean, logvar

# Loss Functions
def vae_loss(x, mean, logvar):
    """Combines reconstruction loss and KL divergence."""
    kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    return tf.reduce_mean(kl_div)

def combined_loss(y_true, reg_pred, clf_pred, mean, logvar, alpha=1.0):
    """Combines regression, classification, and VAE losses."""
    reg_loss = tf.reduce_mean(tf.square(y_true[:, :-1] - reg_pred))
    clf_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true[:, -1:], clf_pred))
    kl_loss = vae_loss(y_true, mean, logvar)
    return reg_loss + clf_loss + alpha * kl_loss

# Example Usage
if __name__ == '__main__':
    input_dim = 5
    latent_dim = 2
    X = np.random.rand(100, input_dim)  # Features
    y = np.hstack([
        np.random.rand(100, 1),         # Regression target
        np.random.randint(0, 2, (100, 1))  # Classification target (binary)
    ])

    model = TwoStageVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training Loop
    for epoch in range(20):
        with tf.GradientTape() as tape:
            reg_pred_stage1, clf_pred_stage1, mean, logvar = model(X, stage=1)
            loss = combined_loss(y, reg_pred_stage1, clf_pred_stage1, mean, logvar, alpha=1.0)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

    # Prediction
    reg_pred, clf_pred, _, _ = model(X, stage=1)
    print("Stage 1 Predictions:", reg_pred[:5], clf_pred[:5])