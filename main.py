import mlflow
import mlflow.keras
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def main():
    # Load the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

    # Define and compile the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Start an MLflow experiment

    # End any active runs
    mlflow.end_run()
    #mlflow.set_experiment("Fashion_MNIST_Image_Classification")

    # Start a run to log parameters and metrics
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("epochs", 2)
        mlflow.log_param("batch_size", 64)

        # Reshape the data for grayscale images
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Train the model
        history = model.fit(x_train, y_train, epochs=10,
                            batch_size=32, validation_data=(x_test, y_test))

        # Log model
        mlflow.keras.log_model(model, "fashion_mnist_model")

        # Log metrics
        mlflow.log_metrics(
            {'final_accuracy': history.history['accuracy'][-1], 'final_loss': history.history['loss'][-1]})

if __name__ == "__main__":
    main()