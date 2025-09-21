# app.py
import streamlit as st
import torch, pickle, os, re
import torch.nn as nn

# ---------- Settings ----------
MODEL_PATH = os.path.join(os.getcwd(), "bilstm_char_model_state.pth")
SRC_VOCAB_PATH = os.path.join(os.getcwd(), "src2id.pkl")
TGT_VOCAB_PATH = os.path.join(os.getcwd(), "tgt2id.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 150

# ---------- Load vocabs ----------
with open(SRC_VOCAB_PATH, "rb") as f:
    src2id = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt2id = pickle.load(f)

id2tgt = {v:k for k,v in tgt2id.items()}

PAD = '<pad>'; BOS = '<bos>'; EOS = '<eos>'; UNK = '<unk>'
# ---------- normalization (same as training) ----------
def normalize_urdu(text):
    text = text or ""
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    text = text.replace('ى','ی')
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s٬،۔؟]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Define same model architecture as training ----------
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=src2id[PAD])
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h_list, c_list = [], []
        for i in range(self.n_layers):
            h_f = h_n[2*i]; h_b = h_n[2*i+1]
            c_f = c_n[2*i]; c_b = c_n[2*i+1]
            h = torch.tanh(self.fc_hidden(torch.cat((h_f, h_b), dim=1)))
            c = c_f + c_b
            h_list.append(h); c_list.append(c)
        h0 = torch.stack(h_list, dim=0); c0 = torch.stack(c_list, dim=0)
        return out, (h0, c0)

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=tgt2id[PAD])
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_tok, hidden, cell):
        emb = self.dropout(self.embedding(input_tok)).unsqueeze(1)
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.0):
        raise NotImplementedError("App only uses encoder/decoder individually for inference.")

# ---------- Instantiate model (use same hyperparams as training) ----------
EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

encoder = BiLSTMEncoder(input_dim=len(src2id), emb_dim=EMB_DIM, hid_dim=HID_DIM,
                       n_layers=ENC_LAYERS, dropout=ENC_DROPOUT)
decoder = LSTMDecoder(output_dim=len(tgt2id), emb_dim=EMB_DIM, hid_dim=HID_DIM,
                      n_layers=DEC_LAYERS, dropout=DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE)

# ---------- Load saved state dict (handle both raw state_dict and dict with 'model_state') ----------
state = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state, dict) and 'model_state' in state:
    model_state = state['model_state']
else:
    model_state = state
# load into encoder+decoder; your saved state may contain only model.state_dict() so this will match
# two-step: try direct load; if fails, try partial load for encoder & decoder keys
try:
    model.load_state_dict(model_state, strict=False)
except Exception:
    # fallback: split keys for encoder/decoder if saved that way (common)
    enc_state = {k.replace('encoder.', ''):v for k,v in model_state.items() if k.startswith('encoder.')}
    dec_state = {k.replace('decoder.', ''):v for k,v in model_state.items() if k.startswith('decoder.')}
    encoder.load_state_dict(enc_state, strict=False)
    decoder.load_state_dict(dec_state, strict=False)

encoder.to(DEVICE); decoder.to(DEVICE)
encoder.eval(); decoder.eval()

# ---------- Greedy decode -------
def greedy_translate(sentence, max_len=MAX_LEN):
    s = normalize_urdu(sentence)
    s_ids = [src2id.get(BOS)] + [src2id.get(ch, src2id[UNK]) for ch in s] + [src2id.get(EOS)]
    if len(s_ids) > max_len:
        s_ids = s_ids[:max_len]; s_ids[-1] = src2id[EOS]
    src_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([len(s_ids)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        _, (h_n, c_n) = encoder(src_tensor, src_lens)
    # adjust decoder layers if needed
    dec_nlayers = decoder.lstm.num_layers
    enc_nlayers = h_n.size(0)
    if enc_nlayers < dec_nlayers:
        last_h = h_n[-1].unsqueeze(0).repeat(dec_nlayers - enc_nlayers, 1, 1)
        last_c = c_n[-1].unsqueeze(0).repeat(dec_nlayers - enc_nlayers, 1, 1)
        h_n = torch.cat([h_n, last_h], dim=0)
        c_n = torch.cat([c_n, last_c], dim=0)
    input_tok = torch.tensor([tgt2id[BOS]], dtype=torch.long).to(DEVICE)
    hidden, cell = h_n, c_n
    out_chars = []
    for _ in range(max_len):
        logits, hidden, cell = decoder(input_tok, hidden, cell)
        top1 = logits.argmax(1).item()
        if top1 == tgt2id[EOS]:
            break
        if top1 != tgt2id[PAD] and top1 != tgt2id[BOS]:
            out_chars.append(id2tgt.get(top1, ''))
        input_tok = torch.tensor([top1], dtype=torch.long).to(DEVICE)
    return ''.join(out_chars)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Urdu → Roman Transliterator", layout="centered")
st.title("📝 Urdu → Roman Urdu Transliterator")
st.write("Enter Urdu text (poetry lines) and click *Transliterate*.")

user_input = st.text_area("Enter Urdu text here:", height=150)
if st.button("Transliterate"):
    text = user_input.strip()
    if not text:
        st.warning("Please enter Urdu text.")
    else:
        pred = greedy_translate(text)
        st.subheader("Roman Urdu")
        st.code(pred)
