import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, target_col='Close', test_size=0.2):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Clean price columns by removing $ and converting to float
    price_cols = ['Close/Last', 'Open', 'High', 'Low']
    for col in price_cols:
        if col in df.columns:
            if df[col].dtype == object:  # Only apply str operations if column contains strings
                df[col] = df[col].str.replace('$', '').astype(float)
            elif not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # If Close column doesn't exist, create it from Close/Last
    if 'Close' not in df.columns and 'Close/Last' in df.columns:
        df['Close'] = df['Close/Last']
    
    # Select features (technical indicators and price data)
    feature_cols = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                   'EMA_5', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 
                   'MACD_Signal', 'MACD_Histogram', 'ATR', 'Sentiment']
    
    # Only use columns that exist in the DataFrame
    features = [col for col in feature_cols if col in df.columns]
    
    # Handle NaN values
    df = df.dropna()
    
    # Split the data into training and testing sets
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler on training data only
    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])
    
    # Create a separate scaler for the target variable for inverse transformation later
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_data[[target_col]])
    
    return train_scaled, test_scaled, train_data, test_data, scaler, target_scaler, features

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target is the closing price (assumed to be at index 0)
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units=units),
        Dropout(dropout),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_evaluate_model(file_path, seq_length=10, epochs=100, batch_size=32):
    # Load and preprocess data
    train_scaled, test_scaled, train_data, test_data, scaler, target_scaler, features = load_and_preprocess_data(file_path)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    # Build the model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions to original scale
    # Create arrays with zeros except for the target column for inverse transformation
    train_predictions_full = np.zeros((len(train_predictions), len(features)))
    train_predictions_full[:, 0] = train_predictions.flatten()
    
    test_predictions_full = np.zeros((len(test_predictions), len(features)))
    test_predictions_full[:, 0] = test_predictions.flatten()
    
    # Inverse transform
    train_predictions = target_scaler.inverse_transform(train_predictions)
    test_predictions = target_scaler.inverse_transform(test_predictions)
    
    # Get actual values
    y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    test_mae = mean_absolute_error(y_test_actual, test_predictions)
    
    train_r2 = r2_score(y_train_actual, train_predictions)
    test_r2 = r2_score(y_test_actual, test_predictions)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_loss_history.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(14, 7))
    
    # Create a date range for the test predictions
    test_dates = test_data.index[seq_length:]
    train_dates = train_data.index[seq_length:]
    
    # Plot training data
    plt.plot(train_dates, y_train_actual, label='Training Actual', alpha=0.5)
    plt.plot(train_dates, train_predictions, label='Training Predicted', alpha=0.5)
    
    # Plot test data
    plt.plot(test_dates, y_test_actual, label='Testing Actual')
    plt.plot(test_dates, test_predictions, label='Testing Predicted')
    
    plt.title('LSTM Model: Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lstm_predictions.png')
    
    return model, history, test_dates, y_test_actual, test_predictions

if __name__ == "__main__":
    file_path = "AAPL_with_indicators.csv"
    
    # Train and evaluate the LSTM model
    model, history, test_dates, actual_values, predicted_values = train_and_evaluate_model(
        file_path,
        seq_length=15,  # Lookback period
        epochs=100,     # Maximum number of epochs (with early stopping)
        batch_size=32   # Batch size
    )
    
    # Save the model
    model.save('lstm_stock_prediction_model.h5')
    print("Model saved as 'lstm_stock_prediction_model.h5'")
    
    # Show plots
    plt.show()
