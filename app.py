import streamlit as st
import pandas as pd
import numpy as np
import re
import textwrap
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import wikipediaapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# ====================== Custom Styling ======================
st.set_page_config(page_title="📚 BookGNN", page_icon="📖", layout="wide")

# Beautiful Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .title {
        font-size: 42px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMetric {
        background: white;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== Header ======================
st.markdown('<h1 class="title">📚 BookGNN Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite book with Graph Neural Networks</p>', unsafe_allow_html=True)

# ====================== Load Models & Data ======================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_data_and_model():
    df = pd.read_csv("book_structural_fingerprints.csv")
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df.get('summary', df['Title']).fillna(df['Title']).astype(str).tolist()
    embeddings = model_sbert.encode(texts, show_progress_bar=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float)

    # Graph Building
    sim_matrix = cosine_similarity(embeddings)
    threshold = 0.65
    edge_index = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim_matrix[i, j] > threshold:
                edge_index.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=embeddings, edge_index=edge_index)

    class BookGNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    device = torch.device('cpu')
    model = BookGNN(384, 256, 384).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(60):   # Reduced for better speed
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        gnn_embeddings = model(data.x, data.edge_index).cpu().numpy()

    return df, model, gnn_embeddings, model_sbert, edge_index, device

df, gnn_model, gnn_embeddings, sbert_model, edge_index, device = load_data_and_model()

st.success(f"✅ Knowledge Base Loaded: **{len(df)} books** | Model Ready!")

# Rest of your functions (fetch_wikipedia_summary, get_full_story_summary, recommend_books) remain same
# ... [I kept your original recommend_books function]

# (For brevity, I'm showing only the UI improvement part. Use your original recommend_books function)

# Chat Interface with better styling
st.markdown("---")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("🔍 Search any book title..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔮 Thinking with Graph Neural Network..."):
            recommend_books(prompt, top_k=5)   # Your original function
            st.session_state.messages.append({"role": "assistant", "content": "✅ Here are the recommendations!"})

# Sidebar with nice design
with st.sidebar:
    st.image("https://source.unsplash.com/300x200/?books", use_column_width=True)
    st.header("About This App")
    st.info("This recommender uses **GraphSAGE + Sentence-BERT** to understand book relationships and suggest similar reads.")
    st.caption("Made with ❤️ using PyTorch Geometric")