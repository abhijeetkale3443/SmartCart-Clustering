
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SmartCart | Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PREMIUM COLOR THEME
# =========================================================
BG = "#0B1120"
SURFACE = "#151F32"
BORDER = "#28364B"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
TEAL = "#14B8A6"
INDIGO = "#818CF8"
AMBER = "#FBBF24"
BLUE = "#60A5FA"

CLUSTER_COLORS = [
    TEAL,
    INDIGO,
    AMBER,
    BLUE,
    "#F472B6",
    "#34D399",
    "#A78BFA",
    "#94A3B8",
]

CHART_TEMPLATE = "plotly_dark"

CATEGORY_LABELS = {
    "MntWines": "Wines",
    "MntFruits": "Fruits",
    "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold Products",
}

CHANNEL_LABELS = {
    "NumWebPurchases": "Web",
    "NumCatalogPurchases": "Catalog",
    "NumStorePurchases": "Store",
    "NumDealsPurchases": "Deals",
}


# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {
        --bg: #0B1120;
        --sidebar: #0F172A;
        --surface: #151F32;
        --border: #28364B;
        --text: #F1F5F9;
        --muted: #94A3B8;
        --teal: #14B8A6;
        --indigo: #818CF8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0B1120;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.94);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.35px;
    }

    p, label {
        color: #CBD5E1;
    }

    /* Main hero */
    .hero {
        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(129,140,248,0.13),
                transparent 32%
            ),
            linear-gradient(
                115deg,
                #142D38 0%,
                #17243A 65%,
                #202344 100%
            );
        border: 1px solid #2A4054;
        border-radius: 18px;
        padding: 27px 29px;
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: #5EEAD4;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #F8FAFC;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(27px, 3vw, 37px);
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #CBD5E1;
        font-size: 14px;
        line-height: 1.7;
        max-width: 760px;
    }

    .hero-pill {
        display: inline-block;
        background: rgba(20,184,166,0.10);
        color: #5EEAD4;
        border: 1px solid rgba(20,184,166,0.30);
        border-radius: 7px;
        padding: 6px 10px;
        margin-top: 14px;
        font-size: 11px;
        font-weight: 700;
    }

    /* Section titles */
    .section-label {
        color: #5EEAD4;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 18px 0 7px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 16px 18px;
        min-height: 110px;
        transition: border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #3A6470;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 12px !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-family: 'Manrope', sans-serif;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Chart cards */
    div[data-testid="stPlotlyChart"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 6px;
        overflow: hidden;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #0F766E;
        color: #FFFFFF;
        border: 1px solid #159E91;
        border-radius: 9px;
        min-height: 40px;
        font-weight: 600;
        transition: background 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #115E59;
        color: #FFFFFF;
        border-color: #5EEAD4;
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input {
        background: #111B2D;
        color: #F1F5F9;
        border-color: #334155;
        border-radius: 9px;
    }

    div[data-baseweb="select"] > div {
        background: #111B2D;
        border-color: #334155;
        border-radius: 9px;
    }

    /* Sidebar navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 5px 0;
        font-size: 13px;
    }

    /* Footer */
    .footer {
        border-top: 1px solid var(--border);
        color: #64748B;
        text-align: center;
        font-size: 11px;
        line-height: 1.8;
        margin-top: 28px;
        padding-top: 20px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        .hero {
            padding: 20px;
        }

        [data-testid="stMetric"] {
            padding: 12px;
            min-height: 95px;
        }

        [data-testid="stMetricValue"] {
            font-size: 21px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def section_heading(title, subtitle=""):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def style_chart(fig, height=350):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color=TEXT,
            size=12,
        ),
        margin=dict(l=16, r=16, t=55, b=16),
        title=dict(
            font=dict(size=15, color=TEXT),
            x=0.03,
            xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color=TEXT,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.12)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.12)",
    )

    return fig


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def segment_color_map(segment_ids):
    return {
        f"Segment {int(cluster_id)}":
            CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
        for cluster_id in segment_ids
    }


# =========================================================
# 5. LOAD, PREPROCESS AND CLUSTER DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_and_cluster(file_bytes, k):
    df = pd.read_csv(BytesIO(file_bytes))

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Fill missing income values
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature engineering
    if "Year_Birth" in df.columns:
        df["Age"] = 2026 - df["Year_Birth"]

    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    available_spending = [
        col for col in spending_columns if col in df.columns
    ]

    if available_spending:
        df["Total_Spending"] = df[available_spending].sum(axis=1)

    if {"Kidhome", "Teenhome"}.issubset(df.columns):
        df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Drop identifier and raw date columns
    df = df.drop(
        columns=[
            col for col in ["ID", "Year_Birth", "Dt_Customer"]
            if col in df.columns
        ],
        errors="ignore",
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable customer records were found.")

    # Fill remaining missing values
    for col in df.select_dtypes(include="number").columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(
            median_value if pd.notna(median_value) else 0
        )

    # Encode categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    X = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int,
    )

    X = X.select_dtypes(include="number")
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] == 0:
        raise ValueError(
            "The dataset has no varying features for clustering."
        )

    if len(X) <= k:
        raise ValueError(
            f"At least {k + 1} customer rows are required."
        )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA dimensionality reduction
    n_components = min(
        3,
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    if n_components < 1:
        raise ValueError("Not enough records to perform PCA.")

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )
    X_pca = pca.fit_transform(X_scaled)

    # K-Means clustering
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(X_pca)

    df["Cluster"] = labels

    # Evaluate clustering
    if 1 < len(np.unique(labels)) < len(df):
        score = silhouette_score(X_pca, labels)
    else:
        score = np.nan

    # PCA coordinates for interactive scatter plot
    coords = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": (
            X_pca[:, 1]
            if X_pca.shape[1] > 1
            else np.zeros(len(df))
        ),
        "Cluster": labels,
    })

    variance = float(
        pca.explained_variance_ratio_.sum() * 100
    )

    return df, coords, score, variance, available_spending


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 18px;">
            <div style="font-size:28px;">🛒</div>
            <div style="font-size:23px;font-weight:800;color:#F8FAFC;">
                SmartCart<span style="color:#14B8A6;">.</span>
            </div>
            <div style="font-size:10px;color:#94A3B8;letter-spacing:1.5px;">
                CUSTOMER INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Analysis Settings")

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Upload the customer dataset you want to analyse.",
    )

    k = st.slider(
        "Number of segments",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Choose how many customer clusters K-Means should create.",
    )

    st.markdown("---")
    st.markdown("### Dashboard")

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Cluster Explorer",
            "Purchase Behaviour",
            "Customer Insights",
            "Dataset Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size:12px;line-height:1.9;color:#94A3B8;">
            <span style="color:#5EEAD4;font-weight:700;">MODEL</span><br>
            K-Means + PCA<br><br>
            <span style="color:#818CF8;font-weight:700;">TECHNOLOGY</span><br>
            Python · Pandas<br>
            Scikit-learn · Plotly<br>
            Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. DATA SOURCE
# =========================================================
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
else:
    file_name = "smartcart_customers.csv"

    try:
        with open(file_name, "rb") as file:
            file_bytes = file.read()
    except FileNotFoundError:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-kicker">CUSTOMER ANALYTICS PLATFORM</div>
                <div class="hero-title">
                    Understand Your Customers.<br>
                    Discover Better Segments.
                </div>
                <div class="hero-description">
                    Explore customer spending, purchasing behaviour and
                    machine-learning segments in one interactive dashboard.
                    Upload your customer CSV from the sidebar to begin.
                </div>
                <div class="hero-pill">
                    ✦ AI-POWERED CUSTOMER SEGMENTATION
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "Dataset not found. Put smartcart_customers.csv beside app.py "
            "or upload your CSV using the sidebar."
        )
        st.stop()


# =========================================================
# 8. RUN THE MODEL
# =========================================================
try:
    with st.spinner("Analysing customer data..."):
        df, coords, score, variance, available_spending = load_and_cluster(
            file_bytes,
            k,
        )
except Exception as error:
    st.error(f"Unable to analyse this dataset: {error}")
    st.stop()


# =========================================================
# 9. SHARED DATA AND METRICS
# =========================================================
cluster_ids = sorted(df["Cluster"].unique())

cluster_names = [
    f"Segment {int(cluster_id)}"
    for cluster_id in cluster_ids
]

colors = segment_color_map(cluster_ids)

df["Segment"] = df["Cluster"].map(
    {
        int(cluster_id): f"Segment {int(cluster_id)}"
        for cluster_id in cluster_ids
    }
)

total_customers = len(df)
total_segments = df["Cluster"].nunique()

average_spending = (
    df["Total_Spending"].mean()
    if "Total_Spending" in df.columns
    else np.nan
)

average_income = (
    df["Income"].mean()
    if "Income" in df.columns
    else np.nan
)

cluster_counts = (
    df["Segment"]
    .value_counts()
    .reindex(cluster_names, fill_value=0)
    .rename_axis("Segment")
    .reset_index(name="Customers")
)


# =========================================================
# 10. MAIN HERO
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">SMARTCART / CUSTOMER ANALYTICS</div>
        <div class="hero-title">
            Customer Intelligence<br>Meets Machine Learning.
        </div>
        <div class="hero-description">
            Transform customer data into meaningful segments. Explore spending
            patterns, purchase channels and customer profiles with an
            interactive machine-learning dashboard.
        </div>
        <div class="hero-pill">✦ K-MEANS CUSTOMER SEGMENTATION</div>
        <div style="margin-top:14px;font-size:11px;color:#94A3B8;">
            SOURCE: {file_name} &nbsp; • &nbsp;
            {total_customers:,} CUSTOMER RECORDS
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 11. KPI CARDS
# =========================================================
section_heading(
    "PERFORMANCE OVERVIEW",
    "A quick snapshot of your customer dataset and segmentation results.",
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Customer Segments", str(total_segments))
m3.metric(
    "Average Spending",
    money(average_spending),
)
m4.metric(
    "Silhouette Score",
    f"{score:.3f}" if pd.notna(score) else "N/A",
    help="Higher scores generally indicate better-separated clusters.",
)

st.markdown("")


# =========================================================
# 12. OVERVIEW PAGE
# =========================================================
if page == "Overview":

    section_heading(
        "CUSTOMER LANDSCAPE",
        "Understand the size and distribution of each segment.",
    )

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        fig = px.pie(
            cluster_counts,
            names="Segment",
            values="Customers",
            hole=0.68,
            color="Segment",
            color_discrete_map=colors,
            title="Customer Segment Distribution",
        )

        fig.update_traces(
            textposition="outside",
            textinfo="percent",
            marker=dict(line=dict(color=BG, width=3)),
        )

        fig.add_annotation(
            text=f"<b>{total_customers:,}</b><br><sup>Customers</sup>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=19, color=TEXT),
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )

    with right:
        fig = px.bar(
            cluster_counts,
            x="Segment",
            y="Customers",
            color="Segment",
            color_discrete_map=colors,
            text="Customers",
            title="Customer Count by Segment",
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )

    section_heading(
        "SPENDING ANALYSIS",
        "Explore the spending distribution across your customer base.",
    )

    if "Total_Spending" in df.columns:
        left, right = st.columns(2, gap="large")

        with left:
            fig = px.histogram(
                df,
                x="Total_Spending",
                nbins=30,
                color_discrete_sequence=[TEAL],
                title="Overall Spending Distribution",
                labels={"Total_Spending": "Total Spending"},
            )

            fig.update_traces(
                marker_line_color=BG,
                marker_line_width=1,
            )

            st.plotly_chart(
                style_chart(fig, 330),
                use_container_width=True,
            )

        with right:
            fig = px.box(
                df,
                x="Segment",
                y="Total_Spending",
                color="Segment",
                color_discrete_map=colors,
                title="Spending by Segment",
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(
                style_chart(fig, 330),
                use_container_width=True,
            )

    else:
        st.info("No recognised spending columns are available.")

    section_heading(
        "SEGMENT SUMMARY",
        "Compare average spending, income and customer activity.",
    )

    agg = {"Cluster": "size"}

    for col in [
        "Total_Spending",
        "Income",
        "Recency",
        "Age",
        "NumWebVisitsMonth",
    ]:
        if col in df.columns:
            agg[col] = "mean"

    summary = df.groupby("Segment").agg(agg).reset_index()

    summary = summary.rename(
        columns={
            "Cluster": "Customers",
            "Total_Spending": "Average Spending",
            "Income": "Average Income",
            "Recency": "Average Recency",
            "Age": "Average Age",
            "NumWebVisitsMonth": "Average Monthly Web Visits",
        }
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 13. CLUSTER EXPLORER PAGE
# =========================================================
elif page == "Cluster Explorer":

    section_heading(
        "CLUSTER EXPLORER",
        "Explore customer groups in the PCA visualization.",
    )

    scatter_df = coords.copy()

    scatter_df["Segment"] = scatter_df["Cluster"].map(
        {
            int(cluster_id): f"Segment {int(cluster_id)}"
            for cluster_id in cluster_ids
        }
    )

    for col in ["Total_Spending", "Income", "Age", "Recency"]:
        if col in df.columns:
            scatter_df[col] = df[col].to_numpy()

    hover_columns = [
        col for col in [
            "Total_Spending",
            "Income",
            "Age",
            "Recency",
        ]
        if col in scatter_df.columns
    ]

    fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color="Segment",
        color_discrete_map=colors,
        hover_data=hover_columns,
        title="PCA Customer Segmentation Map",
        opacity=0.85,
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color=BG))
    )

    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )

    st.plotly_chart(
        style_chart(fig, 480),
        use_container_width=True,
    )

    st.caption(
        f"The selected PCA components capture {variance:.1f}% of the "
        "scaled feature variance. The chart shows projected positions, "
        "not original customer measurements."
    )

    a, b, c = st.columns(3)
    a.metric("Segments Created", total_segments)
    b.metric(
        "Silhouette Score",
        f"{score:.3f}" if pd.notna(score) else "N/A",
    )
    c.metric("PCA Variance Captured", f"{variance:.1f}%")

    section_heading(
        "EXPLORE A SEGMENT",
        "Choose a segment to inspect its customer characteristics.",
    )

    selected_segment = st.selectbox(
        "Customer segment",
        cluster_names,
    )

    selected_cluster = int(selected_segment.split()[-1])

    segment_df = df[df["Cluster"] == selected_cluster]

    a, b, c = st.columns(3)

    a.metric("Customers", f"{len(segment_df):,}")

    b.metric(
        "Average Spending",
        money(segment_df["Total_Spending"].mean())
        if "Total_Spending" in segment_df.columns
        else "N/A",
    )

    c.metric(
        "Average Income",
        money(segment_df["Income"].mean())
        if "Income" in segment_df.columns
        else "N/A",
    )

    profile_columns = [
        col for col in [
            "Age",
            "Income",
            "Total_Spending",
            "Total_Children",
            "Recency",
            "NumWebVisitsMonth",
            "NumWebPurchases",
            "NumStorePurchases",
            "Segment",
        ]
        if col in segment_df.columns
    ]

    st.dataframe(
        segment_df[profile_columns].head(100)
        if profile_columns
        else segment_df.head(100),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 14. PURCHASE BEHAVIOUR PAGE
# =========================================================
elif page == "Purchase Behaviour":

    section_heading(
        "PRODUCT SPENDING",
        "Understand which product categories attract the most spending.",
    )

    if available_spending:

        spending_totals = (
            df[available_spending]
            .sum()
            .rename_axis("Product")
            .reset_index(name="Total Spending")
        )

        spending_totals["Product"] = spending_totals["Product"].map(
            lambda value: CATEGORY_LABELS.get(value, value)
        )

        spending_totals = spending_totals.sort_values(
            "Total Spending",
            ascending=True,
        )

        fig = px.bar(
            spending_totals,
            x="Total Spending",
            y="Product",
            orientation="h",
            color="Total Spending",
            color_continuous_scale=["#164E63", TEAL, INDIGO],
            title="Total Spending by Product Category",
        )

        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True,
        )

        left, right = st.columns(2, gap="large")

        with left:
            avg_spending = (
                df[available_spending]
                .mean()
                .rename_axis("Product")
                .reset_index(name="Average Spending")
            )

            avg_spending["Product"] = avg_spending["Product"].map(
                lambda value: CATEGORY_LABELS.get(value, value)
            )

            fig = px.bar(
                avg_spending,
                x="Product",
                y="Average Spending",
                color_discrete_sequence=[TEAL],
                title="Average Spending per Customer",
            )

            fig.update_layout(xaxis_tickangle=-20)

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )

        with right:
            segment_spending = (
                df.groupby("Segment")[available_spending]
                .mean()
                .reset_index()
                .melt(
                    id_vars="Segment",
                    var_name="Product",
                    value_name="Average Spending",
                )
            )

            segment_spending["Product"] = segment_spending["Product"].map(
                lambda value: CATEGORY_LABELS.get(value, value)
            )

            fig = px.bar(
                segment_spending,
                x="Segment",
                y="Average Spending",
                color="Product",
                barmode="group",
                color_discrete_sequence=CLUSTER_COLORS,
                title="Product Spending by Segment",
            )

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )

    else:
        st.info(
            "Product spending columns were not found in this dataset."
        )

    # Shopping channel behaviour
    section_heading(
        "SHOPPING CHANNELS",
        "Compare online, catalog, store and deal-related purchases.",
    )

    channel_columns = [
        col for col in CHANNEL_LABELS if col in df.columns
    ]

    if channel_columns:

        channel_totals = (
            df[channel_columns]
            .sum()
            .rename_axis("Channel")
            .reset_index(name="Purchases")
        )

        channel_totals["Channel"] = channel_totals["Channel"].map(
            CHANNEL_LABELS
        )

        left, right = st.columns(2, gap="large")

        with left:
            fig = px.bar(
                channel_totals,
                x="Channel",
                y="Purchases",
                color="Channel",
                color_discrete_sequence=[
                    TEAL, INDIGO, BLUE, AMBER
                ],
                title="Total Purchases by Channel",
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(
                style_chart(fig, 340),
                use_container_width=True,
            )

        with right:
            channel_profile = (
                df.groupby("Segment")[channel_columns]
                .mean()
                .reset_index()
                .melt(
                    id_vars="Segment",
                    var_name="Channel",
                    value_name="Average Purchases",
                )
            )

            channel_profile["Channel"] = channel_profile["Channel"].map(
                CHANNEL_LABELS
            )

            fig = px.bar(
                channel_profile,
                x="Segment",
                y="Average Purchases",
                color="Channel",
                barmode="group",
                color_discrete_sequence=[
                    TEAL, INDIGO, BLUE, AMBER
                ],
                title="Average Channel Purchases by Segment",
            )

            st.plotly_chart(
                style_chart(fig, 340),
                use_container_width=True,
            )

    else:
        st.info("No recognised shopping-channel columns were found.")

    # Website visits and recency
    activity_columns = [
        col for col in ["NumWebVisitsMonth", "Recency"]
        if col in df.columns
    ]

    if activity_columns:

        section_heading(
            "CUSTOMER ACTIVITY",
            "Compare website engagement and days since the last purchase.",
        )

        activity = (
            df.groupby("Segment")[activity_columns]
            .mean()
            .reset_index()
            .melt(
                id_vars="Segment",
                var_name="Metric",
                value_name="Average Value",
            )
        )

        activity["Metric"] = activity["Metric"].replace(
            {
                "NumWebVisitsMonth": "Monthly Website Visits",
                "Recency": "Days Since Last Purchase",
            }
        )

        fig = px.bar(
            activity,
            x="Segment",
            y="Average Value",
            color="Metric",
            barmode="group",
            color_discrete_sequence=[TEAL, INDIGO],
            title="Customer Activity by Segment",
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )


# =========================================================
# 15. CUSTOMER INSIGHTS PAGE
# =========================================================
elif page == "Customer Insights":

    section_heading(
        "CUSTOMER INSIGHTS",
        "Filter customers by income and segment to explore their profiles.",
    )

    filtered = df.copy()

    left, right = st.columns(2)

    with left:
        if "Income" in filtered.columns:
            income_min = float(filtered["Income"].min())
            income_max = float(filtered["Income"].max())

            if income_min < income_max:
                income_range = st.slider(
                    "Income range",
                    min_value=income_min,
                    max_value=income_max,
                    value=(income_min, income_max),
                )

                filtered = filtered[
                    filtered["Income"].between(
                        income_range[0],
                        income_range[1],
                    )
                ]
            else:
                st.caption(f"Income value: {money(income_min)}")

    with right:
        selected_segment = st.selectbox(
            "Customer segment",
            ["All Segments"] + cluster_names,
        )

        if selected_segment != "All Segments":
            filtered = filtered[
                filtered["Segment"] == selected_segment
            ]

    st.markdown("")

    a, b, c = st.columns(3)

    a.metric("Matching Customers", f"{len(filtered):,}")

    b.metric(
        "Average Spending",
        money(filtered["Total_Spending"].mean())
        if len(filtered) and "Total_Spending" in filtered.columns
        else "N/A",
    )

    c.metric(
        "Average Income",
        money(filtered["Income"].mean())
        if len(filtered) and "Income" in filtered.columns
        else "N/A",
    )

    display_columns = [
        col for col in [
            "Age",
            "Income",
            "Total_Spending",
            "Total_Children",
            "Recency",
            "NumWebVisitsMonth",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "Segment",
        ]
        if col in filtered.columns
    ]

    section_heading(
        "FILTERED CUSTOMERS",
        "Review the matching customer records or export them to CSV.",
    )

    if filtered.empty:
        st.warning("No customers match the selected filters.")
    else:
        st.dataframe(
            filtered[display_columns]
            if display_columns
            else filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export Filtered Customers CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="smartcart_customer_insights.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Segment profiles
    numeric_columns = [
        col for col in [
            "Age",
            "Income",
            "Total_Spending",
            "Total_Children",
            "Recency",
            "NumWebVisitsMonth",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumDealsPurchases",
        ]
        if col in df.columns
    ]

    if numeric_columns:

        section_heading(
            "SEGMENT PROFILES",
            "Compare average customer characteristics across segments.",
        )

        profile = (
            df.groupby("Segment")[numeric_columns]
            .mean()
            .round(2)
        )

        st.dataframe(
            profile,
            use_container_width=True,
        )

        selected_metric = st.selectbox(
            "Metric to compare",
            numeric_columns,
            index=(
                numeric_columns.index("Total_Spending")
                if "Total_Spending" in numeric_columns
                else 0
            ),
        )

        metric_profile = (
            df.groupby("Segment")[selected_metric]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            metric_profile,
            x="Segment",
            y=selected_metric,
            color="Segment",
            color_discrete_map=colors,
            title=f"Average {selected_metric} by Segment",
            text_auto=".2s",
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )


# =========================================================
# 16. DATASET EXPLORER PAGE
# =========================================================
elif page == "Dataset Explorer":

    section_heading(
        "DATASET EXPLORER",
        "Inspect the customer records and review data quality.",
    )

    a, b, c = st.columns(3)

    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}",
    )

    search_term = st.text_input(
        "Search columns",
        placeholder="Search for Income, Recency, MntWines...",
    )

    visible_df = df.copy()

    if search_term.strip():
        matching_columns = [
            col for col in visible_df.columns
            if search_term.lower() in col.lower()
        ]

        if matching_columns:
            visible_df = visible_df[matching_columns]
        else:
            visible_df = pd.DataFrame()
            st.info("No columns match your search.")

    st.dataframe(
        visible_df,
        use_container_width=True,
        hide_index=True,
    )

    section_heading(
        "DATA QUALITY",
        "Column data types, missing values and unique values.",
    )

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [
            int(value) for value in df.isna().sum().values
        ],
        "Unique Values": [
            int(df[col].nunique()) for col in df.columns
        ],
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Export Clustered Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smartcart_clustered_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 17. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        SMARTCART CUSTOMER INTELLIGENCE<br>
        Built with Python · Streamlit · Plotly · Scikit-learn
        <br>
        Customer segments are model-generated groups and should be
        interpreted alongside business context.
    </div>
    """,
    unsafe_allow_html=True,
)
