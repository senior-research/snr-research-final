import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def prepare_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Clean price columns by removing $ and converting to float
    price_cols = ['Close/Last', 'Open', 'High', 'Low']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].str.replace('$', '').astype(float)
    
    # If Close column doesn't exist, create it from Close/Last
    if 'Close' not in df.columns and 'Close/Last' in df.columns:
        df['Close'] = df['Close/Last']
    
    # Explore correlations with target
    plt.figure(figsize=(12, 10))
    correlation = df.corr()['Close'].sort_values(ascending=False)
    print("Feature correlations with Close price:")
    print(correlation)
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
    return df

def preprocess_data(df, target_col='Close', test_size=0.2, validation_size=0.1):
    # Select features based on correlation analysis
    feature_cols = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                   'EMA_5', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 
                   'MACD_Signal', 'MACD_Histogram', 'ATR', 'Sentiment']
    
    # Only use columns that exist in the DataFrame
    features = [col for col in feature_cols if col in df.columns]
    
    # Handle NaN values
    df = df.dropna()
    
    # Calculate total size
    total_size = len(df)
    train_size = int(total_size * (1 - test_size - validation_size))
    val_size = int(total_size * validation_size)
    
    # Split the data
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size+val_size]
    test_data = df.iloc[train_size+val_size:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler on training data only
    scaler.fit(train_data[features])
    
    train_scaled = scaler.transform(train_data[features])
    val_scaled = scaler.transform(val_data[features])
    test_scaled = scaler.transform(test_data[features])
    
    # Create a separate scaler for the target variable for inverse transformation later
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_data[[target_col]])
    
    return train_scaled, val_scaled, test_scaled, train_data, val_data, test_data, scaler, target_scaler, features

def create_sequences(data, seq_length, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target is the closing price (at the specified index)
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)

def build_bidirectional_lstm_model(input_shape, units=[100, 50], dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    
    # First LSTM layer with return sequences
    model.add(Bidirectional(LSTM(units=units[0], return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(Bidirectional(LSTM(units=units[1])))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, epochs=100, batch_size=32):
    # Build the model
    model = build_bidirectional_lstm_model(input_shape=input_shape)
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler):
    # Make predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    
    # Reshape for inverse transform
    y_train_reshaped = y_train.reshape(-1, 1)
    y_val_reshaped = y_val.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # Inverse transform
    train_predictions = target_scaler.inverse_transform(train_predictions)
    val_predictions = target_scaler.inverse_transform(val_predictions)
    test_predictions = target_scaler.inverse_transform(test_predictions)
    
    y_train_actual = target_scaler.inverse_transform(y_train_reshaped)
    y_val_actual = target_scaler.inverse_transform(y_val_reshaped)
    y_test_actual = target_scaler.inverse_transform(y_test_reshaped)
    
    # Calculate metrics
    metrics = {}
    
    # Training metrics
    metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    metrics['train_mae'] = mean_absolute_error(y_train_actual, train_predictions)
    metrics['train_r2'] = r2_score(y_train_actual, train_predictions)
    
    # Validation metrics
    metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val_actual, val_predictions))
    metrics['val_mae'] = mean_absolute_error(y_val_actual, val_predictions)
    metrics['val_r2'] = r2_score(y_val_actual, val_predictions)
    
    # Test metrics
    metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    metrics['test_mae'] = mean_absolute_error(y_test_actual, test_predictions)
    metrics['test_r2'] = r2_score(y_test_actual, test_predictions)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"Testing RMSE: {metrics['test_rmse']:.4f}")
    
    print(f"\nTraining MAE: {metrics['train_mae']:.4f}")
    print(f"Validation MAE: {metrics['val_mae']:.4f}")
    print(f"Testing MAE: {metrics['test_mae']:.4f}")
    
    print(f"\nTraining R²: {metrics['train_r2']:.4f}")
    print(f"Validation R²: {metrics['val_r2']:.4f}")
    print(f"Testing R²: {metrics['test_r2']:.4f}")
    
    return metrics, train_predictions, val_predictions, test_predictions, y_train_actual, y_val_actual, y_test_actual

def plot_results(history, train_data, val_data, test_data, y_train_actual, y_val_actual, y_test_actual, 
                train_predictions, val_predictions, test_predictions, seq_length):
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('bidirectional_lstm_loss.png')
    plt.close()
    
    # Plot actual vs predicted values
    plt.figure(figsize=(16, 8))
    
    # Create date ranges
    train_dates = train_data.index[seq_length:]
    val_dates = val_data.index[seq_length:]
    test_dates = test_data.index[seq_length:]
    
    # Plot training data
    plt.plot(train_dates, y_train_actual, label='Training Actual', color='blue', alpha=0.5)
    plt.plot(train_dates, train_predictions, label='Training Predicted', color='lightblue', linestyle='--')
    
    # Plot validation data
    plt.plot(val_dates, y_val_actual, label='Validation Actual', color='green', alpha=0.5)
    plt.plot(val_dates, val_predictions, label='Validation Predicted', color='lightgreen', linestyle='--')
    
    # Plot test data
    plt.plot(test_dates, y_test_actual, label='Testing Actual', color='red')
    plt.plot(test_dates, test_predictions, label='Testing Predicted', color='orange', linestyle='--')
    
    plt.title('Bidirectional LSTM Model: Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bidirectional_lstm_predictions.png')
    plt.close()
    
    # Plot test predictions in detail
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_actual, label='Actual', color='blue', marker='o', markersize=3)
    plt.plot(test_dates, test_predictions, label='Predicted', color='red', marker='x', markersize=3)
    plt.title('Test Set Predictions (Detail View)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_predictions_detail.png')
    plt.close()

def main():
    file_path = "AAPL_with_indicators.csv"
    seq_length = 20  # Number of time steps to look back
    
    # Prepare and explore data
    df = prepare_data(file_path)
    
    # Preprocess data
    train_scaled, val_scaled, test_scaled, train_data, val_data, test_data, scaler, target_scaler, features = preprocess_data(df)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_length, target_idx=0)
    X_val, y_val = create_sequences(val_scaled, seq_length, target_idx=0)
    X_test, y_test = create_sequences(test_scaled, seq_length, target_idx=0)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train the model
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        epochs=150,
        batch_size=32
    )
    
    # Evaluate the model
    metrics, train_predictions, val_predictions, test_predictions, y_train_actual, y_val_actual, y_test_actual = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler
    )
    
    # Plot results
    plot_results(
        history, train_data, val_data, test_data,
        y_train_actual, y_val_actual, y_test_actual,
        train_predictions, val_predictions, test_predictions,
        seq_length
    )
    
    # Save the model
    model.save('bidirectional_lstm_stock_model.h5')
    print("\nModel saved as 'bidirectional_lstm_stock_model.h5'")
    
    # Feature importance analysis (future improvement)
    print("\nNext steps for improvement:")
    print("1. Try different architectures (GRU, Transformer)")
    print("2. Hyperparameter tuning")
    print("3. Feature importance analysis")
    print("4. Multi-step forecasting")

if __name__ == "__main__":
    main()
