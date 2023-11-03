import numpy as np
from tqdm import tqdm

from encoder import get_encoder
from tools import get_params


# Multi-head attention

def softmax(x):
    softmax = np.zeros(x.shape)
    for i in range(x.shape[0]):
        e_x = np.exp(x[i] - np.max(x[i]))
        softmax[i] = e_x / e_x.sum()
    return softmax

def attention(Q, K, V):
    s_max = softmax((Q @ K.T)/np.sqrt(K.shape[1]))
    return s_max @ V

def masked_attention(Q, K, V, mask):
    s_max_mask = softmax(((Q @ K.T)/np.sqrt(K.shape[1])) + mask)
    return s_max_mask @ V

def linear_projection(x, w, b):
    return x @ w + b

def multi_head_attention(x, attn, number_of_heads):
    w_1, b_1 = attn["c_attn"]["w"], attn["c_attn"]["b"]
    w_2, b_2 = attn["c_proj"]["w"], attn["c_proj"]["b"]
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    lin1 = linear_projection(x, w_1, b_1)
    Q, K, V = np.split(lin1, 3, axis=1)
    Q_heads = np.split(Q, number_of_heads, axis=1)
    K_heads = np.split(K, number_of_heads, axis=1)
    V_heads = np.split(V, number_of_heads, axis=1)
    for i in range(number_of_heads):
        mask_att = masked_attention(Q_heads[i], K_heads[i], V_heads[i], mask)
        if i == 0:
            merge = mask_att
        else:
            merge = np.concatenate((merge, mask_att), axis=1)
    x = linear_projection(merge, w_2, b_2)
    return x

# Transformer blocks and GPT2

def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * np.power(x, 3))))

def layer_normalization(x, g, b, eps=1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    sigma = np.var(x, axis=1, keepdims=True)
    N = (x - mu)/(np.sqrt(sigma+eps))
    for i in range(N.shape[0]):
        N[i] = g[i]*N[i] + b[i]
    return N

def feed_forward_network(x, mlp):
    w_1, b_1 = mlp["c_fc"]["w"], mlp["c_fc"]["b"]
    w_2, b_2 = mlp["c_proj"]["w"], mlp["c_proj"]["b"]
    lp1 = linear_projection(x, w_1, b_1)
    act = gelu(lp1)
    lp2 = linear_projection(act, w_2, b_2)
    return lp2

def transformer_block(x, block, number_of_heads):
    mlp, attn = block["mlp"], block["attn"]
    ln_1, ln_2 = block["ln_1"], block["ln_2"]
    g_1, b_1, g_2, b_2 = ln_1["g"], ln_1["b"], ln_2["g"], ln_2["b"]
    ln1 = layer_normalization(x, g_1, b_1)
    pass1 = multi_head_attention(ln1, attn, number_of_heads)
    storedx = pass1 + x
    ln2 = layer_normalization(storedx, g_2, b_2)
    fpass = feed_forward_network(ln2, mlp)
    return fpass + storedx


def gpt2(inputs, wte, wpe, blocks, ln_f, number_of_heads):
    g_final, b_final = ln_f["g"], ln_f["b"]
    x = wte[inputs] + wpe[range(len(inputs))]  # Step 1: Sum positional encoding and token encoding
    for i in range(len(blocks)):
        fpass = transformer_block(x, blocks[i], number_of_heads)
        if i == 0:
            merge = fpass
        else:
            merge = np.concatenate((merge, fpass), axis=1)
    ln = layer_normalization(fpass, g_final, b_final)
    return ln @ wte.T

def generate(input_text, tokens_to_generate=40, model_size="124M", models_dir="models", loading_bar=True):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    
    hparams, params = get_params(model_size, models_dir)
    encoder = get_encoder(model_size, models_dir)
    number_of_heads = hparams["n_head"]
    max_context = hparams["n_ctx"]

    # Port the input text to ids
    input_ids = encoder.encode(input_text)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + tokens_to_generate < max_context

    # generate output ids
    output_ids = []
    
    if loading_bar:
        loop_range = tqdm(range(tokens_to_generate), "Thinking...")
    else:
        loop_range = range(tokens_to_generate)

    for _ in loop_range:
        # Call our gtp2 model with input plus generated tokens
        output = gpt2(input_ids + output_ids, **params, number_of_heads=number_of_heads) 

        # Get the next token from the output
        next_id = np.argmax(output[-1])

        # Save the result
        output_ids.append(int(next_id))

    # Port the output ids to text
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    #Test your implemetntation with something like this:
    print(generate("Hello! How do you do?"))    
    """
    You can try out different sized models from this list: ["124M", "355M", "774M", "1558M"]
    Make sure you have enough space on your device since the bigger models are quite large.
    """
    pass
