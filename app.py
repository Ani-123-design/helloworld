
# 🎬 BOLLYWOOD ANALYTICS STREAMLIT DASHBOARD (ADVANCED)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Bollywood Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("BollywoodMovieDetail.csv")
    actors = pd.read_csv("BollywoodActorRanking.csv")
    directors = pd.read_csv("BollywoodDirectorRanking.csv")
    return movies, actors, directors

movies, actors_df, directors_df = load_data()

# =========================
# COLUMN DETECTION
# =========================
def find_col(df, keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

actor_col = find_col(movies, "actor")
director_col = find_col(movies, "director")
genre_col = find_col(movies, "genre")
year_col = find_col(movies, "year")

# =========================
# CLEANING
# =========================
for col in movies.select_dtypes(include=np.number):
    movies[col].fillna(movies[col].mean(), inplace=True)

for col in movies.select_dtypes(exclude=np.number):
    movies[col].fillna(movies[col].mode()[0], inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
movies["rating"] = movies["hitFlop"]
movies["is_hit"] = np.where(movies["hitFlop"] >= 5, 1, 0)
movies["primary_genre"] = movies[genre_col].astype(str).str.split("|").str[0]

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.title("Filters")

selected_genre = st.sidebar.multiselect(
    "Select Genre",
    options=movies["primary_genre"].unique(),
    default=movies["primary_genre"].unique()
)

selected_year = st.sidebar.slider(
    "Select Year Range",
    int(movies[year_col].min()),
    int(movies[year_col].max()),
    (int(movies[year_col].min()), int(movies[year_col].max()))
)

filtered = movies[
    (movies["primary_genre"].isin(selected_genre)) &
    (movies[year_col] >= selected_year[0]) &
    (movies[year_col] <= selected_year[1])
]

# =========================
# HEADER
# =========================
st.title("🎬 Bollywood Movie Analytics Dashboard")

# =========================
# KPI SECTION
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(filtered))
col2.metric("Hit Rate %", round(filtered["is_hit"].mean()*100, 2))
col3.metric("Avg Rating", round(filtered["rating"].mean(), 2))

# =========================
# DISTRIBUTION
# =========================
st.subheader("🎯 Rating Distribution")
fig1 = px.histogram(filtered, x="rating", nbins=20)
st.plotly_chart(fig1, use_container_width=True)

# =========================
# GENRE HIT RATE
# =========================
st.subheader("🎭 Genre vs Hit Rate")
genre_perf = filtered.groupby("primary_genre")["is_hit"].mean().reset_index()
fig2 = px.bar(genre_perf, x="primary_genre", y="is_hit")
st.plotly_chart(fig2, use_container_width=True)

# =========================
# YEAR TREND
# =========================
st.subheader("📈 Hit Rate Over Time")
year_perf = filtered.groupby(year_col)["is_hit"].mean().reset_index()
fig3 = px.line(year_perf, x=year_col, y="is_hit")
st.plotly_chart(fig3, use_container_width=True)

# =========================
# ACTOR-DIRECTOR RELATIONSHIP
# =========================
st.subheader("🤝 Actor-Director Performance")

movies["actor_list"] = movies[actor_col].astype(str).str.split(",")
actor_movie = movies.explode("actor_list")
actor_movie["actor_list"] = actor_movie["actor_list"].str.strip()
actor_movie["director"] = actor_movie[director_col]

combo = actor_movie.groupby(
    ["actor_list", "director"]
).agg(
    rating=("rating","mean"),
    hit_rate=("is_hit","mean"),
    movie_count=("actor_list","count")
).reset_index()

fig4 = px.scatter(
    combo,
    x="movie_count",
    y="rating",
    size="movie_count",
    color="hit_rate",
    hover_data=["actor_list","director"]
)
st.plotly_chart(fig4, use_container_width=True)

# =========================
# 🔥 TOP ACTOR-DIRECTOR DUOS
# =========================
st.subheader("🏆 Top Actor-Director Duos")

top_combo = combo.sort_values(
    ["hit_rate","rating"], ascending=False
).head(10)

fig_duo = px.bar(
    top_combo,
    x="actor_list",
    y="hit_rate",
    color="director",
    hover_data=["movie_count","rating"]
)
st.plotly_chart(fig_duo, use_container_width=True)

# =========================
# 🔥 HITS vs FLOPS BY GENRE
# =========================
st.subheader("🎭 Hits vs Flops by Genre")

genre_split = filtered.groupby("primary_genre").agg(
    hits=("is_hit","sum"),
    total=("is_hit","count")
).reset_index()

genre_split["flops"] = genre_split["total"] - genre_split["hits"]

fig_hf = px.bar(
    genre_split,
    x="primary_genre",
    y=["hits","flops"],
    barmode="stack"
)
st.plotly_chart(fig_hf, use_container_width=True)

# =========================
# 🔥 ACTOR SUCCESS ANALYSIS
# =========================
st.subheader("⭐ Actor Success Analysis")

actor_perf = actor_movie.groupby("actor_list").agg(
    avg_rating=("rating","mean"),
    hit_rate=("is_hit","mean"),
    movie_count=("actor_list","count")
).reset_index()

actor_perf = actor_perf[actor_perf["movie_count"] >= 3]

fig_actor = px.scatter(
    actor_perf,
    x="movie_count",
    y="avg_rating",
    size="movie_count",
    color="hit_rate",
    hover_data=["actor_list"]
)
st.plotly_chart(fig_actor, use_container_width=True)

# =========================
# 🔥 DIRECTOR CONSISTENCY
# =========================
st.subheader("🎬 Director Consistency")

director_perf = filtered.groupby(director_col).agg(
    avg_rating=("rating","mean"),
    hit_rate=("is_hit","mean"),
    movie_count=("rating","count")
).reset_index()

fig_dir = px.scatter(
    director_perf,
    x="movie_count",
    y="avg_rating",
    size="movie_count",
    color="hit_rate",
    hover_data=[director_col]
)
st.plotly_chart(fig_dir, use_container_width=True)

# =========================
# 🔥 GENRE TREND OVER TIME
# =========================
st.subheader("📈 Genre Popularity Over Time")

genre_year = filtered.groupby([year_col,"primary_genre"]).size().reset_index(name="count")

fig_gt = px.line(
    genre_year,
    x=year_col,
    y="count",
    color="primary_genre"
)
st.plotly_chart(fig_gt, use_container_width=True)

# =========================
# 🔥 ACTOR vs GENRE HEATMAP
# =========================
st.subheader("🔥 Actor vs Genre Heatmap")

heat = actor_movie.groupby(["actor_list","primary_genre"]).size().reset_index(name="count")
heat_pivot = heat.pivot(index="actor_list", columns="primary_genre", values="count").fillna(0)

fig_heat = px.imshow(heat_pivot.head(20))
st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# 🔥 RISK vs REWARD
# =========================
st.subheader("⚖️ Risk vs Reward by Genre")

risk = filtered.groupby("primary_genre").agg(
    avg_rating=("rating","mean"),
    volatility=("rating","std")
).reset_index()

fig_rr = px.scatter(
    risk,
    x="volatility",
    y="avg_rating",
    size="avg_rating",
    color="primary_genre"
)
st.plotly_chart(fig_rr, use_container_width=True)

# =========================
# TOP ACTORS (DATASET)
# =========================
st.subheader("⭐ Top Actors (Dataset Ranking)")
top_actors = actors_df.sort_values("normalizedRating", ascending=False).head(10)
fig5 = px.bar(top_actors, x="actorName", y="normalizedRating")
st.plotly_chart(fig5, use_container_width=True)

# =========================
# TOP DIRECTORS (DATASET)
# =========================
st.subheader("🎬 Top Directors (Dataset Ranking)")
top_directors = directors_df.sort_values("normalizedRating", ascending=False).head(10)
fig6 = px.bar(top_directors, x="directorName", y="normalizedRating")
st.plotly_chart(fig6, use_container_width=True)

# =========================
# DATA TABLE
# =========================
st.subheader("📋 Data Preview")
st.dataframe(filtered.head(50))

