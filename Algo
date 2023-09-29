import numpy as np
import tensorflow as tf
import pymysql

# Database connection and data retrieval functions
def connect_to_database():
    connection = pymysql.connect(host='Tensor',
                                 user='root',
                                 password='root',
                                 database='Project',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

def fetch_data_from_db():
    connection = connect_to_database()
    try:
        with connection.cursor() as cursor:
            sql_query = "SELECT temperature, humidity, TRAINING FROM sensor_data"
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

# Fetch data
data = fetch_data_from_db()
data_np = np.array(list(map(lambda row: [row['temperature'], row['humidity']], data)))
labels_np = np.array(list(map(lambda row: row['TRAINING'], data)))

# Normalize or preprocess the data as needed
# For instance:
data_np = (data_np - np.mean(data_np, axis=0)) / np.std(data_np, axis=0)

# Split data into train/test
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data_np, labels_np, test_size=0.2)

# Define the CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),  # We assume 2 features: temperature and humidity
    tf.keras.layers.Reshape((2, 1)),  # Reshape for convolution
    tf.keras.layers.Conv1D(32, kernel_size=1, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Change this if you have more or different output classes
])

model.compile(optimizer='adam',
              loss='mean_squared_error',  # or another suitable loss function
              metrics=['mae'])

# Train the model
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
