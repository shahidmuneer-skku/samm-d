from transformers import Wav2Vec2Model, HubertModel
import torch
import torch.nn as nn

class HubertLSTMClassifier(nn.Module):
    def __init__(self, hubert_model_name, lstm_hidden_size):
        super(HubertLSTMClassifier, self).__init__()
        # self.hubert = HubertModel.from_pretrained(hubert_model_name)
        self.lstm = nn.LSTM(input_size=1024,
                            hidden_size=lstm_hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(2*lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_values):
        # with torch.no_grad():
        #     features = self.hubert(input_values).last_hidden_state
        lstm_out, _ = self.lstm(input_values)
        # print(lstm_out[:, -1, :].shape)
        logits = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output for classification
        return self.sigmoid(logits)

# Example usage
