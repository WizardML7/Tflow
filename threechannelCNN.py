import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = 'Wednesday-workingHours.pcap_ISCX.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces

# Check for 'Label' column
if 'Label' not in df.columns:
    raise KeyError("Column 'Label' is missing from the DataFrame.")

# Drop columns with missing values (optional, depending on your dataset)
df.dropna(axis=1, inplace=True)

# Encode categorical labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Split features and target
X = df.drop(['Label'], axis=1)  # Features
y = df['Label']  # Target

# Check for and handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

print(df.columns)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check the shape of the data
print("X_train shape before reshaping:", X_train.shape)

# Let's assume we want to use 3 channels and duplicate the data across channels
num_channels = 3

# If you have only one feature, you can duplicate it across channels
X_train = np.repeat(X_train[:, :, np.newaxis], num_channels, axis=2)  # (samples, features, channels)
X_test = np.repeat(X_test[:, :, np.newaxis], num_channels, axis=2)

# Check the shape after reshaping
print("X_train shape after reshaping:", X_train.shape)

# One-hot encode the target labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Build a Convolutional Neural Network model for 3 channels
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], num_channels)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


