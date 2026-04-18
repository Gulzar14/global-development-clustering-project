import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt

# --------------------
# LOAD MODELS
# --------------------
kmeans_model = pickle.load(open("kmeans_model.pkl","rb"))
dbscan_model = pickle.load(open("dbscan_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
pca = pickle.load(open("pca.pkl","rb"))

# --------------------
# LOAD DATA
# --------------------
data = pd.read_excel("World_development_mesurement.xlsx")

# Keep original for display
display_data = data.copy()

# --------------------
# CLEAN DATA
# --------------------
model_data = data.copy()

for col in model_data.columns:
    if col != "Country":
        model_data[col] = model_data[col].astype(str)
        model_data[col] = model_data[col].str.replace('%','',regex=False)
        model_data[col] = model_data[col].str.replace('$','',regex=False)
        model_data[col] = model_data[col].str.replace(',','',regex=False)
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

numeric_data = model_data.drop("Country",axis=1)
numeric_data = numeric_data.fillna(numeric_data.median())
numeric_data = numeric_data.fillna(0)
model_data.update(numeric_data)

# --------------------
# SIDEBAR
# --------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["KMeans", "DBSCAN"])

country1 = st.sidebar.selectbox("Country 1", display_data["Country"].unique())
country2 = st.sidebar.selectbox("Country 2", display_data["Country"].unique(), index=10)

# --------------------
# SELECT MODEL
# --------------------
if model_choice == "KMeans":
    model = kmeans_model
else:
    model = dbscan_model

# --------------------
# GET ROWS
# --------------------
row1_display = display_data[display_data["Country"] == country1].iloc[0]
row1_model = model_data[model_data["Country"] == country1].iloc[[0]]

row2_display = display_data[display_data["Country"] == country2].iloc[0]
row2_model = model_data[model_data["Country"] == country2].iloc[[0]]

# --------------------
# TRANSFORM & PREDICT (Do this once)
# --------------------
scaled_all = scaler.transform(numeric_data)
reduced_all = pca.transform(scaled_all)

if model_choice == "KMeans":
    clusters = kmeans_model.predict(reduced_all)
else:
    clusters = dbscan_model.fit_predict(reduced_all)


model_data["Cluster"] = clusters
display_data["Cluster"] = clusters

# --------------------
# GET COUNTRY DATA
# --------------------

def get_country_cluster(c_name):
    return model_data[model_data["Country"] == c_name]["Cluster"].values[0]

cluster1 = get_country_cluster(country1)
cluster2 = get_country_cluster(country2)


def cluster_name(c):
    if c == 0: return "Developed"
    elif c == 1: return "Developing"
    elif c == 2: return "Underdeveloped"
    elif c == -1: return "Outlier"
    else: return "Unknown"

# --------------------
# COUNTRY COMPARISON UI
# --------------------
st.header("Country Comparison")
col1, col2 = st.columns(2)


row1_display = display_data[display_data["Country"] == country1].iloc[0]
row2_display = display_data[display_data["Country"] == country2].iloc[0]

for col, row_display, cluster, name in [
    (col1, row1_display, cluster1, country1),
    (col2, row2_display, cluster2, country2)
]:
    with col:
        st.subheader(name)
        st.success(f"Classification: {cluster_name(cluster)}")

        gdp_val = str(row_display["GDP"]).replace("$","").replace(",","")
        gdp_val = pd.to_numeric(gdp_val, errors="coerce")
        
        st.metric("GDP", f"${gdp_val:,.0f}" if pd.notna(gdp_val) else "N/A")
        st.metric("Birth Rate", row_display["Birth Rate"])
        st.metric("Internet Usage", row_display["Internet Usage"])
# --------------------
# GLOBAL SUMMARY
# --------------------
st.header("Global Development Indicators")

scaled_all = scaler.transform(numeric_data)
reduced_all = pca.transform(scaled_all)
if model_choice == "KMeans":
    clusters = model.predict(reduced_all)
else:
    clusters = dbscan_model.fit_predict(reduced_all)

total_countries = len(model_data["Country"].unique())
developed = sum(clusters == 0)
developing = sum(clusters == 1)
underdeveloped = sum(clusters == 2)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Countries", total_countries)
c2.metric("Developed", developed)
c3.metric("Developing", developing)
c4.metric("Underdeveloped", underdeveloped)

# --------------------
# TABLE
# --------------------
st.header("Country Indicators")
st.dataframe(display_data[display_data["Country"].isin([country1,country2])])

# --------------------
# BAR CHART
# --------------------
st.header("Cluster Distribution")
counts = pd.Series(clusters).value_counts().sort_index()
fig,ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Countries")
st.pyplot(fig)

# --------------------
# RANKING
# --------------------
st.header("Country Development Rankings")

ranking_df = pd.DataFrame({
    "Country": model_data["Country"],
    "Cluster": clusters,
    "GDP": pd.to_numeric(
        model_data["GDP"].astype(str).str.replace("$","").str.replace(",",""),
        errors="coerce"
    )
})

ranking_df = ranking_df.dropna(subset=["GDP"])

top_dev = ranking_df[ranking_df["Cluster"] == 0].sort_values("GDP",ascending=False).head(10)
top_under = ranking_df[ranking_df["Cluster"] == 2].sort_values("GDP",ascending=True).head(10)

colA,colB = st.columns(2)

with colA:
    st.subheader("Top Developed Countries")
    fig1 = px.bar(top_dev, x="Country", y="GDP", color="GDP")
    st.plotly_chart(fig1)

with colB:
    st.subheader("Least Developed Countries")
    fig2 = px.bar(top_under, x="Country", y="GDP", color="GDP")
    st.plotly_chart(fig2)

# --------------------
# MAP
# --------------------
st.header("Global Development Map")
map_data = pd.DataFrame({
    "Country": model_data["Country"],
    "Cluster": clusters
})

fig = px.choropleth(
    map_data,
    locations="Country",
    locationmode="country names",
    color="Cluster",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig)
