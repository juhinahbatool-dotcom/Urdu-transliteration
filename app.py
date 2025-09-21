import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# ====================
# Load vocab + model
# ====================

# Load vocabs
with open("src2id.pkl", "rb") as f:
    src2id = pickle.load(f)
with open("tgt2id.pkl", "rb") as f:
    tgt2id = pickle.load(f)

id2tgt = {v: k for k, v in tgt2id.items()}

# Special tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

MAX_LEN = 150
DEVICE = torch.device("cpu")

# Model definition (same as training, but smaller copy)
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=src2id[PAD])
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, src, src_lens):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        h_list, c_list = [], []
        for i in range(self.n_layers):
            h_f = h_n[2*i]
            h_b = h_n[2*i+1]
            c_f = c_n[2*i]
            c_b = c_n[2*i+1]
            h = torch.tanh(self.fc_hidden(torch.cat((h_f, h_b), dim=1)))
            c = c_f + c_b
            h_list.append(h)
            c_list.append(c)
        h0 = torch.stack(h_list, dim=0)
        c0 = torch.stack(c_list, dim=0)
        return (h0, c0)

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=tgt2id[PAD])
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_tok, hidden, cell):
        emb = self.embedding(input_tok).unsqueeze(1)
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens):
        return self.encoder(src, src_lens)

# ====================
# Load model
# ====================
EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT = 0.3

enc = BiLSTMEncoder(len(src2id), EMB_DIM, HID_DIM, ENC_LAYERS, DROPOUT)
dec = LSTMDecoder(len(tgt2id), EMB_DIM, HID_DIM, DEC_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

model.load_state_dict(torch.load("bilstm_char_model_state.pth", map_location=DEVICE))
model.eval()

# ====================
# Helpers
# ====================
def normalize_urdu(text):
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)    # remove tashkeel
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    text = text.replace('ى','ی')
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s٬،۔؟]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def greedy_translate(sentence, max_len=MAX_LEN):
    s = normalize_urdu(sentence)
    s_ids = [src2id.get(BOS)] + [src2id.get(ch, src2id[UNK]) for ch in s] + [src2id.get(EOS)]
    src_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([len(s_ids)], dtype=torch.long).to(DEVICE)

    h_n, c_n = model.encoder(src_tensor, src_lens)

    input_tok = torch.tensor([tgt2id[BOS]], dtype=torch.long).to(DEVICE)
    hidden, cell = h_n, c_n
    out_chars = []
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_tok, hidden, cell)
        top1 = logits.argmax(1).item()
        if top1 == tgt2id[EOS]:
            break
        if top1 not in (tgt2id[PAD], tgt2id[BOS]):
            out_chars.append(id2tgt[top1])
        input_tok = torch.tensor([top1], dtype=torch.long).to(DEVICE)
    return ''.join(out_chars)

# ====================
# Streamlit UI
# ====================
st.title("🌐 Urdu → Roman Transliteration")
st.write("Enter Urdu text below and get Roman Urdu transliteration:")

user_input = st.text_area("Your Urdu text:", "")

if st.button("Transliterate"):
    if user_input.strip():
        result = greedy_translate(user_input)
        st.success(result)
    else:
        st.warning("Please enter some Urdu text.")