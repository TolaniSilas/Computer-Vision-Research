"""
ViT building blocks: PatchEmbed (Embedded Patches), Transformer Encoder, Multi-Head Attention, MLP.

Reference: "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).
https://arxiv.org/pdf/2010.11929 
"""
import copy
import torch
import torch.nn as nn


class MLP(nn.Module):
    """multi-layer perceptron for the vision transformer (ViT)."""

    def __init__(self, input_size, hidden_size, output_size):
        """initializes the mlp with two linear layers."""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # two-layer mlp with gelu activation.
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Dropout(p=0.5),
        )

        # initialize weights after layers are defined.
        self._init_weights()  


    def _init_weights(self):
        """initializes weights using xavier uniform and near-zero biases."""

        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.normal_(module.bias, std=1e-6)


    def forward(self, x):
        """applies the multi-layer perceptron to input."""

        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """multi-head attention mechanism with multiple parallel attention heads for the vision transformer (ViT)."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """initialize the multi-head attention layer."""

        super().__init__()

        # ensure output dimension is divisible by number of heads.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # store dimensions.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # query projection layer.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

        # key projection layer.
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)

        # value projection layer.
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # output projection to combine heads.
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout layer.
        self.dropout = nn.Dropout(dropout)

        # causal mask to prevent attending to future positions (in order to avoid cheating).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        """applies multi-head attention to input sequence."""

        # get batch size, sequence length, and input dimension.
        b, num_tokens, d_in = x.shape

        # compute queries, keys, and values.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split into multiple heads by reshaping.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # rearrange to (batch, num_heads, seq_len, head_dim).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute attention scores.
        attn_scores = queries @ keys.transpose(2, 3)

        # apply causal mask to prevent attending to future tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # get key dimension for scaling.
        d_k = keys.shape[-1]
        scaling_factor = d_k**0.5

        # apply scaled softmax to get attention weights.
        attn_weights = torch.softmax(attn_scores / scaling_factor, dim=-1)

        # apply dropout to attention weights.
        attn_weights = self.dropout(attn_weights)

        # compute weighted sum of values.
        context_vec = attn_weights @ values

        # rearrange back to (batch, seq_len, num_heads, head_dim).
        context_vec = context_vec.transpose(1, 2)

        # combine all heads by concatenating.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # apply output projection.
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    """transformer block for the vision transformer (ViT)."""

    def __init__(self, config):
        """initializes the transformer block with a multi-head attention layer and a mlp layer."""
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config["embed_dim"])
        self.multi_head_attention = MultiHeadAttention(config["num_heads"], config["head_dim"])
        self.mlp = MLP(config["embed_dim"], config["hidden_size"], config["embed_dim"])
        self.norm2 = nn.LayerNorm(config["embed_dim"])

    def forward(self, x):
        """applies the transformer block to the input."""

        # extract the input.
        input = x

        # apply layer normalization to the input.
        x = self.layer_norm1(x)

        # apply multi-head attention to the normalized input.
        x = self.multi_head_attention(x)

        # add the output of the multi-head attention to the original input (perform a residual connection).
        x = x + input

        # extract the updated input (it will be used for residual connection).
        input = x

        # apply layer normalization to the output of the multi-head attention.
        x = self.norm2(x)

        # apply the mlp with two layers to the normalized output of the residual connection (mhsa output + original input).
        x = self.mlp(x)

        # add the mlp output to the residual connection of multi-head attention.
        x = x + input

        return x  


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.trfblock = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        for i in range(config["num_layers"]):
            layer = TransformerBlock(config)
            self.trfblock.append(copy.deepcopy(layer))

  

    def forward(self, hidden_states):

        for layer_block in self.trfblock:
            hidden_states= layer_block(hidden_states)
        
        encoded = self.encoder_norm(hidden_states)

        return encoded


class PatchEmbed(nn.Module):
    """embedding patches for the vision transformer (ViT)."""

    def __init__(self):
        """initializes the patch embedding with a linear layer."""
        super().__init__()






class TransformerEncoder(nn.Module):
    """transformer encoder for the vision trasnformer (ViT)."""

    def __init__(self, config, img_size):
        """initializes the transformer encoder with a transformer block (multi-head attention and mlp layers) and embedding patches layer."""
        super().__init__()

        self.patch_embed = PatchEmbed(config, img_size=img_size)
        self.transformer_blocks = EncoderBlock(config)

    
    def forward(self, x):
        """applies the transformer encoder to the input."""
        
        patch_embed_output = self.patch_embed(x)

        encoded = self.transformer_blocks(patch_embed_output)
        
        return encoded