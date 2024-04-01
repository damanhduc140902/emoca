from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow import keras

def build_nn_model(input_shape):
    """
    Xây dựng mô hình mạng nơ-ron sâu.

    Args:
        input_shape (tuple): Kích thước đầu vào cho mô hình.
    
    Returns:
        keras.models.Sequential: Mô hình mạng nơ-ron đã được xây dựng.
    """
    model = Sequential()
    model.add(Dense(256, activation='tanh', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # Số lớp đầu ra dựa trên số lượng nhãn

    optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
