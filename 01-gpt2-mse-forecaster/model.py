from transformers import GPT2Model
import torch
from torch import nn, Tensor


class GPTEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_tokens, dropout_prob = 0.2):
        super(GPTEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv1d(16, 128, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=1)
        self.sos_embedding = nn.Linear(num_tokens, out_channels)

        self.relu = nn.ReLU()

    def forward(self, src: Tensor, sos: Tensor) -> Tensor:

        src = src.permute(0, 2, 1)
        # Conv Embedding
        src = self.conv1(src)
        src = self.dropout1(src)
        src = self.relu(src)
        src = self.conv2(src)
        src = self.dropout2(src)
        src = self.relu(src)
        src = self.conv3(src)
        src = self.relu(src)                

        # linear embedding
        if sos is not None:
            sos = self.sos_embedding(sos)
            sos = self.relu(sos)
            sos = sos.unsqueeze(1).permute(0,2,1)
            # print(src.shape, sos.shape)
            src = torch.cat((sos, src), dim=2)
        src = src.permute(0,2,1)
        return src

class GPTOutput(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob = 0.2):
        super(GPTOutput, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(128, 16, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.conv3 = nn.Conv1d(16, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, src: Tensor) -> Tensor:

        src = src.permute(0, 2, 1)
        # Conv output
        src = self.conv1(src)
        src = self.dropout1(src)
        src = self.relu(src)
        src = self.conv2(src)
        src = self.dropout2(src)
        src = self.relu(src)
        src = self.conv3(src)
        src = src.permute(0,2,1)
        return src

class StockGPT2Model(nn.Module):
    def __init__(self, num_features, num_tickers):
        super(StockGPT2Model, self).__init__()
        # Keep the rest of the original model (without the embedding layer)
        
        self.gpt2 = GPT2Model.from_pretrained("gpt2", resid_pdrop = 0, embd_pdrop = 0,
                                 attn_pdrop = 0, n_positions = 1024,
                                 n_layer = 4, n_head = 6, n_embd = 252,
                                 n_inner = 504, ignore_mismatched_sizes=True)
        embd_dim = self.gpt2.config.n_embd
        self.embedding = GPTEmbedding(in_channels = num_features, out_channels = embd_dim, num_tokens=num_tickers)
        self.output = GPTOutput(in_channels = embd_dim, out_channels = num_features)

    def forward(self, src: Tensor, sos = None, past_key_values = None) -> Tensor:
        # Apply your custom embedding
        embedding = self.embedding(src, sos)
        # gpt2
        gpt2_output = self.gpt2(inputs_embeds=embedding, past_key_values=past_key_values)
        #output
        tgt = self.output(gpt2_output.last_hidden_state)
        return tgt, gpt2_output.past_key_values

