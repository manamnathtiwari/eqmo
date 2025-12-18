import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    LSTM baseline for stress prediction.
    Standard sequence-to-sequence LSTM for time series forecasting.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: (batch, seq_len, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Decode each timestep
        output = self.fc(lstm_out)  # (batch, seq_len, output_size)
        
        return output
