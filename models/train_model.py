# logger
import logging
from config import M, T, df

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prepare model's layers

def create_model():
    # TODO: add doc and hyperparameters tuning
    # TODO: rework the architecture of the model
    model = Sequential()
    model.add(Dense(12, input_shape=(T,), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    logger.info(model.summary())

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, verbose=True):
    logger.info("Training model...")
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_data=(X_test, y_test), verbose=verbose)
    logger.info("Done.")
    return history
