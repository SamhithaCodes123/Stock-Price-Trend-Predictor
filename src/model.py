import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        
        return out

class StockPredictor:
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StockLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = None
        
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, patience=15):
        """Train the LSTM model"""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).view(-1, 1).to(self.device)
        
        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            # Batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = self.criterion(test_outputs, y_test).item()
            
            avg_train_loss = epoch_loss / (len(X_train) // batch_size + 1)
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            
            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'temp_best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('temp_best_model.pth'))
        os.remove('temp_best_model.pth')
        
        return train_losses, test_losses
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def save_model(self, filepath, feature_info=None):
        """Save model and metadata"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_params': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
            }
        }
        
        if feature_info:
            save_dict['feature_info'] = feature_info
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and metadata"""
        # checkpoint = torch.load(filepath, map_location=self.device)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        
        # Create model with saved parameters
        params = checkpoint['model_params']
        self.model = StockLSTM(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers']
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        feature_info = checkpoint.get('feature_info', None)
        print(f"Model loaded from {filepath}")
        
        return feature_info
