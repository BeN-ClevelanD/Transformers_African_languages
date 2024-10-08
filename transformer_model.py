import torch
import torch.nn as nn
import math
# The following code is a PyTorch implementation of the Transformer model

# Class for creating fixed implementation of the positional encodings to be added to the input embeddings in the Transformer model
class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the paper
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    This encoding adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the PositionalEncoding module.
        
        Args:
            d_model (int): The dimension of the model's hidden states.
            max_len (int): The maximum sequence length to support.
        """
        super().__init__()
        # Initialize the positional encoding matrix with zeros
        pe = torch.zeros(max_len, d_model)
        # Create a tensor with following shape :  (max_len, 1),  contaiining position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Below Computes the division term used in the sine and cosine functions
        #  term makes sure  that dif dimensions use dif frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # below applies  the sine function to even indices of the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        #Below applies cos to odd positions in enc matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension by unsqueezing and then transpose to ensure the shape is (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #Register the pos enc matrix as a buffer (not trainable)
        self.register_buffer('pe', pe)
    #ADd pos encodings (each position respectively) To the input embeddings in the forward pass method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
        
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

#Below defines the Transformer model class implementation
class TransformerModel(nn.Module):
    """
    Implements a Transformer model as described in the paper
    "Attention Is All You Need" (Vaswani et al., 2017) with some modifications.
    """
    #Constructor method for the Transformer model class , initialize with defaults
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, dropout: float = 0.1, layer_norm_position: str = 'after', 
                 weight_tying: bool = False, positional_encoding: str = 'learnt'):
        """
        Initialize the TransformerModel.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model's hidden states.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            layer_norm_position (str): Position of layer normalization ('before' or 'after').
            weight_tying (bool): Whether to tie input and output embeddings.
            positional_encoding (str): Type of positional encoding ('fixed' or 'learnt').
        """
        super().__init__()
        self.d_model = d_model
        # Embedding layer to convert input tokens to dense vectors
        self.embed = nn.Embedding(vocab_size, d_model)
        # Conditional Positional Encoding: either fixed or learnt embeddings, depending on config
        if positional_encoding == 'fixed':
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            self.pos_encoder = nn.Embedding(5000, d_model)
        #BElow sets up the transformers encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=(layer_norm_position == 'before'))
        #Stack as many layers as specified in the config according to num_layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        #Fully connected layer to output the predictions using vocab
        self.fc_out = nn.Linear(d_model, vocab_size)
        #Assign dropout layer according to config
        self.dropout = nn.Dropout(dropout)
        #Weight tying on or off depending on config
        if weight_tying:
            self.fc_out.weight = self.embed.weight
        # Initialize weights of the embedding and output layers
        self.init_weights()
#function to initialize the weights of the embedding and output layers
    def init_weights(self):
        """Initialize the weights of the embedding and output layers."""
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    #Forward pass method for the Transformer model
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            src_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Apply embeddings to the input chars/ tokens and scale by the square root of d_model/hidden statee size
        src = self.embed(src) * math.sqrt(self.d_model)
        #Conditionally apply positional encodings to the input embeddings
        if isinstance(self.pos_encoder, PositionalEncoding):
            src = self.pos_encoder(src)
        else:
            #below sets up for learnt pos embeddings
            positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
            #add the learnt pos encodings to the input embeddings
            src = src + self.pos_encoder(positions)
        #Apply dropout to the input embeddings
        src = self.dropout(src)
        #Apply the transformer encoder layers to the input embeddings
        output = self.transformer_encoder(src, src_mask)
        #Apply dropout to the output of the transformer encoder layers
        output = self.dropout(output)
        #return the generated predictions from output
        return self.fc_out(output)