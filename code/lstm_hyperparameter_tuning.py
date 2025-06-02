import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import itertools
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, target_col='Close'):
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
    
    # Select features
    feature_cols = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
                   'EMA_5', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 
                   'MACD_Signal', 'MACD_Histogram', 'ATR', 'Sentiment']
    
    # Only use columns that exist in the DataFrame
    features = [col for col in feature_cols if col in df.columns]
    
    # Handle NaN values
    df = df.dropna()
    
    # Split data: 70% train, 15% validation, 15% test
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size+val_size]
    test_data = df.iloc[train_size+val_size:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler on training data only
    train_scaled = scaler.fit_transform(train_data[features])
    val_scaled = scaler.transform(val_data[features])
    test_scaled = scaler.transform(test_data[features])
    
    # Create a separate scaler for the target variable for inverse transformation later
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_data[[target_col]])
    
    return train_scaled, val_scaled, test_scaled, train_data, val_data, test_data, scaler, target_scaler, features

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target is the closing price (assumed to be at index 0)
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_model(model_type, input_shape, units, dropout_rate, learning_rate):
    model = Sequential()
    
    if model_type == 'lstm':
        # Simple LSTM model
        model.add(LSTM(units=units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units[1]))
        model.add(Dropout(dropout_rate))
    
    elif model_type == 'bidirectional_lstm':
        # Bidirectional LSTM model
        model.add(Bidirectional(LSTM(units=units[0], return_sequences=True), input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units=units[1])))
        model.add(Dropout(dropout_rate))
    
    elif model_type == 'gru':
        # GRU model
        model.add(GRU(units=units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units=units[1]))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_and_evaluate(X_train, y_train, X_val, y_val, model_type, units, dropout_rate, 
                       learning_rate, batch_size, epochs, seq_length):
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model
    model = build_model(model_type, input_shape, units, dropout_rate, learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Get the best validation loss
    best_val_loss = min(history.history['val_loss'])
    
    # Number of epochs the model actually trained for
    epochs_trained = len(history.history['loss'])
    
    return {
        'model_type': model_type,
        'units': units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'val_loss': best_val_loss,
        'epochs_trained': epochs_trained,
        'training_time': training_time
    }

def hyperparameter_tuning():
    # Load and preprocess data
    file_path = "AAPL_with_indicators.csv"
    
    # Different sequence lengths to try
    sequence_lengths = [10, 20, 30]
    results = []
    
    for seq_length in sequence_lengths:
        print(f"\nTuning for sequence length: {seq_length}")
        
        # Load and preprocess with this sequence length
        train_scaled, val_scaled, test_scaled, train_data, val_data, test_data, scaler, target_scaler, features = (
            load_and_preprocess_data(file_path)
        )
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_val, y_val = create_sequences(val_scaled, seq_length)
        
        # Define hyperparameter grid
        param_grid = {
            'model_type': ['lstm', 'bidirectional_lstm', 'gru'],
            'units': [[50, 25], [100, 50]],
            'dropout_rate': [0.2, 0.3],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32],
            'epochs': [100]  # With early stopping
        }
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        print(f"Testing {len(combinations)} hyperparameter combinations...")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            
            print(f"Combination {i+1}/{len(combinations)}: {params}")
            
            # Train and evaluate
            result = train_and_evaluate(
                X_train, y_train, X_val, y_val, 
                params['model_type'], params['units'], params['dropout_rate'],
                params['learning_rate'], params['batch_size'], params['epochs'],
                seq_length
            )
            
            results.append(result)
            
            # Print intermediate result
            print(f"  Val Loss: {result['val_loss']:.6f}, Time: {result['training_time']:.2f}s")
    
    # Sort by validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    # Print top 5 results
    print("\nTop 5 Hyperparameter Combinations:")
    for i, result in enumerate(results[:5]):
        print(f"\nRank {i+1}:")
        print(f"  Model Type: {result['model_type']}")
        print(f"  Sequence Length: {result['seq_length']}")
        print(f"  Units: {result['units']}")
        print(f"  Dropout Rate: {result['dropout_rate']}")
        print(f"  Learning Rate: {result['learning_rate']}")
        print(f"  Batch Size: {result['batch_size']}")
        print(f"  Validation Loss: {result['val_loss']:.6f}")
        print(f"  Epochs Trained: {result['epochs_trained']} (with early stopping)")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
    
    # Get the best combination
    best_result = results[0]
    
    # Train and evaluate final model with the best parameters
    train_final_model(best_result)
    
    return results

def train_final_model(best_params):
    print("\nTraining final model with best parameters...")
    
    # Load and preprocess data
    file_path = "AAPL_with_indicators.csv"
    seq_length = best_params['seq_length']
    
    train_scaled, val_scaled, test_scaled, train_data, val_data, test_data, scaler, target_scaler, features = (
        load_and_preprocess_data(file_path)
    )
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(
        best_params['model_type'], 
        input_shape, 
        best_params['units'], 
        best_params['dropout_rate'], 
        best_params['learning_rate']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # More epochs for final model
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    
    # Inverse transform
    y_test_reshaped = y_test.reshape(-1, 1)
    test_predictions_transformed = target_scaler.inverse_transform(test_predictions)
    y_test_transformed = target_scaler.inverse_transform(y_test_reshaped)
    
    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test_transformed, test_predictions_transformed))
    print(f"\nTest RMSE with best model: {test_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    
    # Create date range
    test_dates = test_data.index[seq_length:]
    
    # Plot test predictions
    plt.plot(test_dates, y_test_transformed, label='Actual', color='blue')
    plt.plot(test_dates, test_predictions_transformed, label='Predicted', color='red')
    
    plt.title(f'Best Model Test Predictions - {best_params["model_type"].upper()}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('best_model_predictions.png')
    
    # Plot loss history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Best Model Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('best_model_loss.png')
    
    # Save model
    model.save(f'best_{best_params["model_type"]}_model.h5')
    print(f"Model saved as 'best_{best_params['model_type']}_model.h5'")
    
    # Save best parameters to a file
    with open('best_parameters.txt', 'w') as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTest RMSE: {test_rmse:.4f}")
    
    print("Best parameters saved to 'best_parameters.txt'")

if __name__ == "__main__":
    # Run hyperparameter tuning
    results = hyperparameter_tuning()
