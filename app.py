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
st.set_page_config(page_title="📚 BookGNN Recommender", page_icon="📖", layout="wide")
st.title("📚 GNN-based Topological Book Recommender (GraphSAGE)")
st.markdown("**Exact same logic as your Notebook** with full evaluation metrics")

# ====================== Load Models & Data ======================
@st.cache_resource(show_spinner="Loading SBERT + Building Graph + Training GNN...")
def load_data_and_model():
    df = pd.read_csv("book_structural_fingerprints.csv")
    
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

    texts = df.get('summary', df['Title']).fillna(df['Title']).astype(str).tolist()
    embeddings = model_sbert.encode(texts, show_progress_bar=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float)

    # Build Graph
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    model.train()
    for epoch in range(80):
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

st.success(f"✅ Knowledge Base Loaded ({len(df)} books) | GNN Training Completed")

# ====================== Wikipedia ======================
def fetch_wikipedia_summary(title):
    try:
        wiki = wikipediaapi.Wikipedia(user_agent="BookGNNRecommender/1.0", language='en')
        queries = [f"{title} book", f"{title} novel", title]
        for q in queries:
            page = wiki.page(q)
            if page.exists() and len(page.summary) > 100:
                return page.summary
        return None
    except:
        return None

def get_full_story_summary(title):
    full = fetch_wikipedia_summary(title)
    if not full or len(full.strip()) < 30:
        return "Full summary not available on Wikipedia."
    cleaned = re.sub(r'\s+', ' ', full).strip()
    return textwrap.fill(cleaned, width=100)

# ====================== Main Recommendation Function (Exact Logic) ======================
def recommend_books(user_title, top_k=5):
    match = df[df['Title'].str.lower() == user_title.lower().strip()]

    if not match.empty:
        idx = match.index[0]
        query_emb = gnn_embeddings[idx].reshape(1, -1)
        is_in_dataset = True
        st.success(f"✨ Found '{user_title}' in local dataset.")
    else:
        st.info(f"🔍 '{user_title}' not in dataset. Computing on-the-fly...")
        summary = fetch_wikipedia_summary(user_title)
        if not summary:
            st.error("❌ Could not find summary for this title.")
            return

        new_emb = sbert_model.encode([summary])
        new_emb = torch.tensor(new_emb, dtype=torch.float).to(device)

        with torch.no_grad():
            projected_emb = gnn_model(new_emb, edge_index[:2, :0])
            query_emb = projected_emb.cpu().numpy()

        is_in_dataset = False

    # Similarity
    similarities = cosine_similarity(query_emb, gnn_embeddings)[0]
    top_indices = similarities.argsort()[-top_k-10:][::-1]

    # ====================== Evaluation Metrics (Same as Notebook) ======================
    if is_in_dataset:
        valid_sims = [(i, sim) for i, sim in enumerate(similarities) if i != idx]
        if valid_sims:
            closest_idx, max_sim = max(valid_sims, key=lambda x: x[1])
            closest_features = gnn_embeddings[closest_idx].reshape(1, -1)
        else:
            max_sim = 0.0
            closest_features = query_emb
    else:
        closest_idx = similarities.argmax()
        max_sim = similarities.max()
        closest_features = gnn_embeddings[closest_idx].reshape(1, -1)

    mse = mean_squared_error(query_emb, closest_features)
    var = np.var(gnn_embeddings)
    error_rate = mse / var if var > 0 else 0.0
    accuracy = (max_sim * 100)
    if max_sim > 0.85:
        accuracy = min(98.0, accuracy * 1.1)

    confidence = "High" if accuracy >= 85 else "Medium" if accuracy >= 65 else "Low"

    # Show Metrics like Notebook
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Best Similarity", f"{max_sim:.4f}")
    with col2: st.metric("Structural Accuracy", f"{accuracy:.2f}%")
    with col3: st.metric("MSE", f"{mse:.4f}")
    with col4: st.metric("Confidence", confidence)

    # ====================== Show Recommendations ======================
    st.subheader(f"📖 TOP RECOMMENDATIONS for **{user_title}**")
    count = 0
    for i in top_indices:
        if is_in_dataset and i == idx:
            continue

        rec_title = df.iloc[i]['Title']
        sim_score = similarities[i]
        full_summary = get_full_story_summary(rec_title)

        st.write(f"**{count+1}. {rec_title}**")
        st.write(f"Similarity: `{sim_score:.3f}`")
        st.write(full_summary)
        st.divider()

        count += 1
        if count >= top_k:
            break

    if count == 0:
        st.warning("No recommendations found.")

# ====================== Chat Interface ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter the Book Title:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching using GraphSAGE..."):
            recommend_books(prompt, top_k=5)
            st.session_state.messages.append({"role": "assistant", "content": "✅ Recommendations generated."})

with st.sidebar:
    st.header("About")
    st.write("This is **exact port** of your notebook code into Streamlit.")
    st.caption("First run takes time due to model training.")