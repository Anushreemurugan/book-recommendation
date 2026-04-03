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

# ====================== Page Config ======================
st.set_page_config(page_title="BookGNN Recommender", page_icon="📖", layout="wide")

# Clean & Modern Styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 3rem 0;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .subtitle {
        font-size: 22px;
        opacity: 0.95;
        font-weight: 400;
    }
    .rec-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 6px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== Header ======================
st.markdown("""
    <div class="header">
        <h1 class="title">📚 Book Recommender</h1>
        <p class="subtitle">Discover your next favorite book using Topological Recommendations</p>
    </div>
""", unsafe_allow_html=True)

# ====================== Load Models ======================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_data_and_model():
    df = pd.read_csv("book_structural_fingerprints.csv")
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df.get('summary', df['Title']).fillna(df['Title']).astype(str).tolist()
    embeddings = model_sbert.encode(texts, show_progress_bar=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float)

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
    for epoch in range(50):
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

st.success(f"✅ Loaded **{len(df)} books** • GNN Model Ready")

# ====================== Wikipedia Functions ======================
def fetch_wikipedia_summary(title):
    try:
        wiki = wikipediaapi.Wikipedia(user_agent="BookGNN/1.0", language='en')
        for q in [f"{title} book", f"{title} novel", title]:
            page = wiki.page(q)
            if page.exists() and len(page.summary) > 100:
                return page.summary
        return None
    except:
        return None

def get_full_story_summary(title):
    full = fetch_wikipedia_summary(title)
    if not full or len(full.strip()) < 30:
        return None
    cleaned = re.sub(r'\s+', ' ', full).strip()
    return textwrap.fill(cleaned, width=100)

# ====================== Recommendation with Fallback ======================
def recommend_books(user_title, top_k=5):
    match = df[df['Title'].str.lower() == user_title.lower().strip()]

    if not match.empty:
        idx = match.index[0]
        query_emb = gnn_embeddings[idx].reshape(1, -1)
        is_in_dataset = True
    else:
        st.info(f"🔍 Computing embedding for **{user_title}**...")
        summary = fetch_wikipedia_summary(user_title)
        if not summary:
            st.error("❌ Could not find information for this title.")
            return
        new_emb = sbert_model.encode([summary])
        new_emb = torch.tensor(new_emb, dtype=torch.float).to(device)
        with torch.no_grad():
            projected = gnn_model(new_emb, edge_index[:2, :0])
            query_emb = projected.cpu().numpy()
        is_in_dataset = False

    similarities = cosine_similarity(query_emb, gnn_embeddings)[0]
    top_indices = similarities.argsort()[-top_k-15:][::-1]   # Extra candidates

    recommendations = []
    fallback_books = []

    for i in top_indices:
        if is_in_dataset and i == idx:
            continue
            
        rec_title = df.iloc[i]['Title']
        sim_score = similarities[i]
        summary_text = get_full_story_summary(rec_title)

        if summary_text:
            recommendations.append((rec_title, sim_score, summary_text))
        else:
            fallback_books.append((rec_title, sim_score, "Full summary not available on Wikipedia."))

        if len(recommendations) >= top_k:
            break

    # Fill with fallback if needed
    while len(recommendations) < top_k and fallback_books:
        recommendations.append(fallback_books.pop(0))

    st.subheader(f"📖 Top Recommendations for **{user_title}**")

    for count, (rec_title, sim_score, summary_text) in enumerate(recommendations):
        st.markdown(f"""
        <div class="rec-card">
            <h4>⭐ {count+1}. {rec_title}</h4>
            <p><strong>Similarity:</strong> {sim_score:.3f}</p>
            <p>{summary_text}</p>
        </div>
        """, unsafe_allow_html=True)

    if len(recommendations) < top_k:
        st.info(f"Showing {len(recommendations)} recommendations with available summaries.")

# ====================== Chat Interface ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("🔍 Enter a book title (e.g., Dune, The Alchemist, Atomic Habits)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding similar books with available summaries..."):
            recommend_books(prompt, top_k=5)

with st.sidebar:
    st.image("https://source.unsplash.com/400x250/?books,library", use_column_width=True)
    st.header("About BookGNN")
    st.info("This recommender uses **Graph Neural Networks (GraphSAGE)** to understand deep topological relationships between books.")
    st.caption("Built with ❤️ using PyTorch Geometric & Streamlit")
