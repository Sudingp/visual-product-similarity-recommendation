"""
Amazon-Style Visual Product Similarity & Image-Based Recommendation System.
Interactive Streamlit Application.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Amazon Visual Search & Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Amazon-inspired Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        color: #131921;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 25px;
    }
    .amazon-badge {
        background-color: #FF9900;
        color: #111;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 6px;
    }
    .match-badge {
        background-color: #232f3e;
        color: #ff9900;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        display: inline-block;
        margin-bottom: 4px;
    }
    .product-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0F1111;
        line-height: 1.25;
        height: 2.6em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-top: 6px;
    }
    .price-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #B12704;
    }
    .rating-text {
        font-size: 0.85rem;
        color: #007185;
    }
    .card-container {
        border: 1px solid #E3E6E6;
        border-radius: 8px;
        padding: 12px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .latency-pill {
        background-color: #e7f4ea;
        color: #0d6832;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.recommender import VisualProductRecommender


@st.cache_resource(show_spinner="Loading ResNet-50 backbone & FAISS Index...")
def get_recommender():
    """Initializes and caches the recommendation engine for zero-latency re-runs."""
    return VisualProductRecommender(
        metadata_path="data/metadata.csv",
        index_path="models/faiss_index.bin",
        embeddings_path="models/product_embeddings.npy"
    )


def render_stars(rating: float) -> str:
    """Returns visual star icons for rating."""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars


def main():
    # Header Banner
    st.markdown('<h1 class="main-header">🛒 Visual Product Similarity & Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Amazon-Style Deep Learning Visual Search using <b>ResNet-50 Embeddings</b> and <b>FAISS Vector Indexing</b></p>', unsafe_allow_html=True)

    # Check if artifacts exist
    if not (os.path.exists("models/faiss_index.bin") and os.path.exists("data/metadata.csv")):
        st.error("⚠️ Model index or catalog metadata not found! Please run `python src/build_catalog_index.py` first.")
        st.stop()

    recommender = get_recommender()
    metadata_df = recommender.metadata_df
    categories = ["All Categories"] + sorted(metadata_df['category'].unique().tolist())

    # Sidebar Controls & Filters
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=130)
        st.markdown("### 🔍 Search & Recommendation Filters")

        top_k = st.slider("Number of Recommendations (Top-K)", min_value=3, max_value=18, value=6, step=3)

        category_filter = st.selectbox(
            "Category Filter",
            options=categories,
            index=0,
            help="Filter recommendations to a specific product category or keep across all categories."
        )

        min_price, max_price = st.slider(
            "Price Range ($ USD)",
            min_value=float(metadata_df['price_usd'].min()),
            max_value=float(metadata_df['price_usd'].max()),
            value=(float(metadata_df['price_usd'].min()), float(metadata_df['price_usd'].max())),
            step=10.0
        )

        min_rating = st.select_slider(
            "Minimum Customer Rating",
            options=[0.0, 3.5, 4.0, 4.2, 4.5, 4.8],
            value=0.0,
            format_func=lambda x: "Any Rating" if x == 0.0 else f"{x}+ Stars"
        )

        st.markdown("---")
        st.markdown("### 📊 System Specs")
        st.markdown(f"- **Catalog Size:** `{len(metadata_df):,}` products")
        st.markdown(f"- **Embedding Backbone:** `ResNet50` (2048-dim)")
        st.markdown(f"- **Vector Engine:** `FAISS IndexFlatIP`")
        st.markdown(f"- **Metric:** Cosine Similarity")

    # Main Navigation Tabs
    tab_search, tab_browse, tab_evaluation = st.tabs([
        "📷 Reverse Image Search",
        "🛍️ Browse Catalog ('Customers Also Viewed')",
        "📈 Quantitative Evaluation & Metrics"
    ])

    # ---------------- TAB 1: REVERSE IMAGE SEARCH ----------------
    with tab_search:
        st.markdown("#### Upload any product photo to find visually identical or similar items:")
        col_upload, col_preview = st.columns([1.5, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload a query image (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a photo from your phone or web"
            )

        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            with col_preview:
                st.image(query_image, caption="Uploaded Query Image", use_container_width=True)

            if st.button("🚀 Find Visually Similar Products", type="primary"):
                with st.spinner("Extracting ResNet50 visual embeddings and querying FAISS..."):
                    recommendations, latency_ms, _ = recommender.recommend(
                        query_input=query_image,
                        top_k=top_k,
                        category_filter=category_filter,
                        min_price=min_price,
                        max_price=max_price,
                        min_rating=min_rating if min_rating > 0 else None,
                        exclude_self=False
                    )

                st.markdown(f"""
                <div style="margin: 15px 0;">
                    <span class="latency-pill">⚡ FAISS Query Time: <b>{latency_ms:.2f} ms</b></span>
                    &nbsp;&nbsp; Retrieved <b>{len(recommendations)}</b> visually similar products
                </div>
                """, unsafe_allow_html=True)

                if len(recommendations) == 0:
                    st.warning("No items matched your specific filter constraints. Try expanding the price range or category filter.")
                else:
                    cols = st.columns(3)
                    for idx, row in recommendations.iterrows():
                        col_idx = idx % 3
                        with cols[col_idx]:
                            with st.container():
                                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                                if os.path.exists(row['local_image_path']):
                                    st.image(row['local_image_path'], use_container_width=True)
                                else:
                                    st.image("https://via.placeholder.com/300?text=Product+Image", use_container_width=True)
                                
                                st.markdown(f'<span class="match-badge">🎯 {row["similarity_pct"]}% Visual Match</span>', unsafe_allow_html=True)
                                st.markdown(f'<span class="amazon-badge">Rank #{row["rank"]}</span>', unsafe_allow_html=True)
                                st.markdown(f'<div class="product-title" title="{row["title"]}">{row["title"]}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="rating-text">{render_stars(row["rating"])} ({row["review_count"]:,})</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="price-text">${row["price_usd"]:.2f}</div>', unsafe_allow_html=True)
                                st.button(f"Add to Cart", key=f"btn_upload_{row['image_id']}_{idx}", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.write("")

    # ---------------- TAB 2: BROWSE CATALOG ----------------
    with tab_browse:
        st.markdown("#### Select any product from the catalog to see Amazon-style 'Similar Items':")

        col_cat_sel, col_prod_sel = st.columns(2)
        with col_cat_sel:
            browse_cat = st.selectbox("1. Filter Catalog by Category:", options=categories[1:], index=0)
        
        filtered_cat_items = metadata_df[metadata_df['category'] == browse_cat]

        with col_prod_sel:
            selected_title = st.selectbox(
                "2. Choose a Product:",
                options=filtered_cat_items['title'].tolist()
            )

        selected_row = filtered_cat_items[filtered_cat_items['title'] == selected_title].iloc[0]
        selected_idx = selected_row.name

        col_detail, col_recs = st.columns([1, 2.2])

        with col_detail:
            st.markdown("### 📌 Selected Product")
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            if os.path.exists(selected_row['local_image_path']):
                st.image(selected_row['local_image_path'], use_container_width=True)
            st.markdown(f"**{selected_row['title']}**")
            st.markdown(f"Category: `{selected_row['category']}`")
            st.markdown(f"<span class='rating-text'>{render_stars(selected_row['rating'])} ({selected_row['review_count']} reviews)</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='price-text'>${selected_row['price_usd']:.2f}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_recs:
            st.markdown("### 💡 Customers Who Viewed This Item Also Viewed")
            recommendations, latency_ms, _ = recommender.recommend(
                query_input=selected_idx,
                top_k=top_k,
                category_filter=category_filter,
                min_price=min_price,
                max_price=max_price,
                min_rating=min_rating if min_rating > 0 else None,
                exclude_self=True
            )

            st.markdown(f'<span class="latency-pill">⚡ FAISS Latency: <b>{latency_ms:.2f} ms</b></span>', unsafe_allow_html=True)
            st.write("")

            cols = st.columns(3)
            for idx, row in recommendations.iterrows():
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    if os.path.exists(row['local_image_path']):
                        st.image(row['local_image_path'], use_container_width=True)
                    st.markdown(f'<span class="match-badge">🎯 {row["similarity_pct"]}% Match</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="product-title" title="{row["title"]}">{row["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="rating-text">{render_stars(row["rating"])} ({row["review_count"]:,})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="price-text">${row["price_usd"]:.2f}</div>', unsafe_allow_html=True)
                    st.button("View Product", key=f"btn_cat_{row['image_id']}_{idx}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.write("")

    # ---------------- TAB 3: QUANTITATIVE EVALUATION ----------------
    with tab_evaluation:
        st.markdown("### 📈 Quantitative Retrieval & Latency Benchmarks")
        st.markdown("""
        Visual similarity retrieval performance evaluated across the catalog using **Precision@K** and **Recall@K**,
        where ground-truth relevance is determined by the item's verified product class and category.
        """)

        eval_json_path = "data/evaluation_results.json"
        if os.path.exists(eval_json_path):
            import json
            with open(eval_json_path, "r") as f:
                eval_data = json.load(f)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Precision @ 1", eval_data["k_metrics"]["K=1"]["precision_at_k_pct"])
            with col_m2:
                st.metric("Precision @ 5", eval_data["k_metrics"]["K=5"]["precision_at_k_pct"])
            with col_m3:
                st.metric("Precision @ 10", eval_data["k_metrics"]["K=10"]["precision_at_k_pct"])
            with col_m4:
                st.metric("Mean Latency", f"{eval_data['latency']['mean_ms']:.3f} ms")

            # Benchmark Table
            st.markdown("#### Detailed Metric Breakdown Across Values of K:")
            table_rows = []
            for k_label, vals in eval_data["k_metrics"].items():
                table_rows.append({
                    "Top-K Cutoff": k_label,
                    "Precision@K": vals["precision_at_k_pct"],
                    "Recall@K": vals["recall_at_k_pct"]
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            st.markdown("#### Vector Engine Latency Distribution (FAISS CPU):")
            lat = eval_data["latency"]
            st.write(f"- **Mean Latency:** `{lat['mean_ms']:.3f} ms`")
            st.write(f"- **Median (P50):** `{lat['p50_ms']:.3f} ms`")
            st.write(f"- **95th Percentile (P95):** `{lat['p95_ms']:.3f} ms`")
            st.write(f"- **99th Percentile (P99):** `{lat['p99_ms']:.3f} ms`")
        else:
            st.info("Run `python src/evaluate.py` to generate the complete quantitative benchmark results.")


if __name__ == "__main__":
    main()
