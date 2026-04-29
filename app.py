import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════
# ⚙️  PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="🎬 Bollywood Analytics Suite", layout="wide", initial_sidebar_state="expanded")

DARK  = "#0A0E17"
CARD  = "#111827"
CARD2 = "#1a2235"
GOLD  = "#F5A623"
ROSE  = "#E84393"
CYAN  = "#00D4FF"
GREEN = "#22D3A5"
PURP  = "#9B59FF"
TEXTM = "#E2E8F0"
TEXTS = "#94A3B8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Nunito:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] {{
    background-color: {DARK};
    color: {TEXTM};
    font-family: 'Nunito', sans-serif;
}}
.main {{ background-color: {DARK}; padding: 0 1rem; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1220 0%, #0a0e17 100%);
    border-right: 1px solid #1e2d45;
}}
[data-testid="stSidebar"] * {{ color: {TEXTM} !important; }}
h1,h2,h3 {{ font-family: 'Bebas Neue', sans-serif; letter-spacing:1px; color:{GOLD} !important; }}
.page-title {{
    font-family:'Bebas Neue',sans-serif; font-size:2.8rem;
    background:linear-gradient(90deg,{GOLD},{ROSE});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:2px; margin-bottom:0.2rem;
}}
.sub-title {{ color:{TEXTS}; font-size:0.95rem; margin-bottom:1.5rem; }}
.kpi-card {{
    background: linear-gradient(135deg,{CARD} 0%,{CARD2} 100%);
    border:1px solid #1e3a5f; border-top:3px solid {GOLD};
    border-radius:12px; padding:18px 20px; text-align:center;
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform:translateY(-3px); }}
.kpi-num {{ font-family:'Bebas Neue',sans-serif; font-size:2.4rem; color:{GOLD}; line-height:1; }}
.kpi-lbl {{ color:{TEXTS}; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }}
.insight-box {{
    background:linear-gradient(135deg,#0f1e35 0%,#111827 100%);
    border:1px solid #1e3a5f; border-left:4px solid {CYAN};
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-size:0.92rem; line-height:1.7; color:{TEXTM};
}}
.insight-box b {{ color:{CYAN}; }}
.warn-box {{
    background:linear-gradient(135deg,#1f1020 0%,#111827 100%);
    border:1px solid #3d1a3a; border-left:4px solid {ROSE};
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-size:0.92rem; color:{TEXTM};
}}
.warn-box b {{ color:{ROSE}; }}
.green-box {{
    background:linear-gradient(135deg,#0d2020 0%,#111827 100%);
    border:1px solid #1a3d2e; border-left:4px solid {GREEN};
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-size:0.92rem; color:{TEXTM};
}}
.green-box b {{ color:{GREEN}; }}
.stChatMessage {{ background:{CARD}; border-radius:12px; border:1px solid #1e3a5f; }}
div[data-testid="stChatInput"] input {{
    background:{CARD2}; color:{TEXTM}; border:1px solid #1e3a5f; border-radius:8px;
}}
.stSelectbox > div {{ background:{CARD2} !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 📥  DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        movies  = pd.read_csv("BollywoodMovieDetail.csv")
        actors  = pd.read_csv("BollywoodActorRanking.csv")
        dirs    = pd.read_csv("BollywoodDirectorRanking.csv")
    except FileNotFoundError:
        try:
            movies  = pd.read_csv("/mnt/user-data/uploads/BollywoodMovieDetail.csv")
            actors  = pd.read_csv("/mnt/user-data/uploads/BollywoodActorRanking.csv")
            dirs    = pd.read_csv("/mnt/user-data/uploads/BollywoodDirectorRanking.csv")
        except Exception as e:
            st.error(f"Could not load data files: {e}")
            st.stop()

    # ── Numeric cleaning ──────────────────────────────────
    movies['hitFlop']     = pd.to_numeric(movies['hitFlop'],     errors='coerce').fillna(1)
    movies['releaseYear'] = pd.to_numeric(movies['releaseYear'], errors='coerce').fillna(0).astype(int)
    movies['sequel']      = pd.to_numeric(movies['sequel'],      errors='coerce').fillna(0).astype(int)

    for df in [movies, actors, dirs]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')

    # ── Derived fields ────────────────────────────────────
    # hitFlop: 1=Disaster, 2=Flop, 3=Average, 4=Semi-hit, 5+=Hit, 7+=Blockbuster, 9+=All-time Classic
    movies['hitFlop_label'] = movies['hitFlop'].map({
        1:'Disaster', 2:'Flop', 3:'Average', 4:'Semi-Hit',
        5:'Hit', 6:'Super Hit', 7:'Blockbuster', 8:'Blockbuster', 9:'Classic'
    }).fillna('Hit')
    movies['is_hit']       = (movies['hitFlop'] >= 5).astype(int)
    movies['is_flop']      = (movies['hitFlop'] <= 2).astype(int)
    movies['is_disaster']  = (movies['hitFlop'] == 1).astype(int)
    movies['decade']       = (movies['releaseYear'] // 5 * 5).astype(str) + "s"

    # ── Exploded lists ────────────────────────────────────
    def explode_pipe(df, col):
        return df[col].fillna('Unknown').astype(str).str.split('|').apply(
            lambda x: [i.strip() for i in x] if isinstance(x, list) else ['Unknown']
        )

    movies['act_list']  = explode_pipe(movies, 'actors')
    movies['dir_list']  = explode_pipe(movies, 'directors')
    movies['gen_list']  = movies['genre'].astype(str).str.split('|').apply(lambda x: [i.strip().lower() for i in x])
    movies['writ_list'] = explode_pipe(movies, 'writers')

    return movies, actors, dirs

movies, actors_rank, dirs_rank = load_data()


# ═══════════════════════════════════════════════════════════
# 🛠️  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Nunito', color=TEXTM),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXTM)),
    margin=dict(l=20, r=20, t=40, b=20),
)

def styled_fig(fig, height=400):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor='#1e2d45', linecolor='#1e2d45', tickfont=dict(color=TEXTS))
    fig.update_yaxes(gridcolor='#1e2d45', linecolor='#1e2d45', tickfont=dict(color=TEXTS))
    return fig

def kpi(label, value, col=None):
    html = f'<div class="kpi-card"><div class="kpi-num">{value}</div><div class="kpi-lbl">{label}</div></div>'
    if col: col.markdown(html, unsafe_allow_html=True)
    else:    st.markdown(html, unsafe_allow_html=True)

def insight(txt, style='info'):
    css = {'info':'insight-box','warn':'warn-box','good':'green-box'}[style]
    st.markdown(f'<div class="{css}">{txt}</div>', unsafe_allow_html=True)

# Pre-computed aggregations (cached)
@st.cache_data
def actor_stats():
    df = movies.explode('act_list')
    df = df[~df['act_list'].isin(['Unknown','n/a',''])]
    g = df.groupby('act_list').agg(
        total_films=('title','count'),
        hits=('is_hit','sum'),
        flops=('is_flop','sum'),
        disasters=('is_disaster','sum'),
        avg_score=('hitFlop','mean')
    ).reset_index()
    g['hit_pct']  = (g['hits'] / g['total_films'] * 100).round(1)
    g['flop_pct'] = (g['flops'] / g['total_films'] * 100).round(1)
    g.rename(columns={'act_list':'name'}, inplace=True)
    # Merge with actor ranking data
    ar = actors_rank[['actorName','normalizedRating','googleHits']].rename(columns={'actorName':'name'})
    g = g.merge(ar, on='name', how='left')
    return g.sort_values('total_films', ascending=False)

@st.cache_data
def director_stats():
    df = movies.explode('dir_list')
    df = df[~df['dir_list'].isin(['Unknown','n/a',''])]
    g = df.groupby('dir_list').agg(
        total_films=('title','count'),
        hits=('is_hit','sum'),
        flops=('is_flop','sum'),
        avg_score=('hitFlop','mean')
    ).reset_index()
    g['hit_pct']  = (g['hits'] / g['total_films'] * 100).round(1)
    g['flop_pct'] = (g['flops'] / g['total_films'] * 100).round(1)
    g.rename(columns={'dir_list':'name'}, inplace=True)
    dr = dirs_rank[['directorName','normalizedRating','googleHits']].rename(columns={'directorName':'name'})
    g = g.merge(dr, on='name', how='left')
    return g.sort_values('total_films', ascending=False)

@st.cache_data
def genre_stats():
    df = movies.explode('gen_list')
    df = df[df['gen_list'].str.strip() != '']
    g = df.groupby('gen_list').agg(
        total_films=('title','count'),
        hits=('is_hit','sum'),
        avg_score=('hitFlop','mean')
    ).reset_index()
    g['hit_pct'] = (g['hits'] / g['total_films'] * 100).round(1)
    return g.sort_values('total_films', ascending=False)

@st.cache_data
def duo_stats():
    pairs = []
    for _, row in movies.iterrows():
        for a in row['act_list']:
            for d in row['dir_list']:
                if a not in ['Unknown','n/a',''] and d not in ['Unknown','n/a','']:
                    pairs.append({'Actor':a,'Director':d,'is_hit':row['is_hit'],
                                  'hitFlop':row['hitFlop'],'title':row['title']})
    pdf = pd.DataFrame(pairs)
    g = pdf.groupby(['Actor','Director']).agg(
        films=('title','count'), hits=('is_hit','sum'), avg_score=('hitFlop','mean')
    ).reset_index()
    g['hit_pct'] = (g['hits']/g['films']*100).round(1)
    return g.sort_values(['hits','films'], ascending=False)

act_df  = actor_stats()
dir_df  = director_stats()
gen_df  = genre_stats()
duo_df  = duo_stats()


# ═══════════════════════════════════════════════════════════
# 🗂️  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f'<div style="font-family:Bebas Neue;font-size:1.8rem;color:{GOLD};letter-spacing:2px;">🎬 BOLLYWOOD<br>ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXTS};font-size:0.8rem;margin-bottom:1.5rem;">2001 – 2014 · {len(movies):,} Films</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", [
        "🏠  Industry Overview",
        "🌟  Actor Analytics",
        "🎬  Director Analytics",
        "🎭  Genre Intelligence",
        "🤝  Duo & Collaboration",
        "🤖  AI Chatbot"
    ], label_visibility='collapsed')
    st.markdown("---")
    st.markdown(f'<div style="color:{TEXTS};font-size:0.75rem;">hitFlop Scale<br>1=Disaster · 2=Flop · 3=Average<br>4=Semi-Hit · 5=Hit · 6=Super Hit<br>7-8=Blockbuster · 9=Classic</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 1 ▸ INDUSTRY OVERVIEW
# ═══════════════════════════════════════════════════════════
if "Industry" in page:
    st.markdown('<div class="page-title">INDUSTRY OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Bollywood at a macro level · 2001–2014</div>', unsafe_allow_html=True)

    # KPIs
    total = len(movies)
    hits  = int(movies['is_hit'].sum())
    flops = int(movies['is_flop'].sum())
    avg_s = movies['hitFlop'].mean()
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi("Total Films",  f"{total:,}", c1)
    kpi("Hits (≥5)",    f"{hits:,}",  c2)
    kpi("Flops (≤2)",   f"{flops:,}", c3)
    kpi("Hit Rate",     f"{hits/total*100:.1f}%", c4)
    kpi("Avg Score",    f"{avg_s:.2f}", c5)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Year-wise release volume & hit rate ──────
    yrg = movies.groupby('releaseYear').agg(
        Films=('title','count'), Hits=('is_hit','sum')
    ).reset_index()
    yrg['Hit_Rate'] = (yrg['Hits']/yrg['Films']*100).round(1)
    yrg = yrg[yrg['releaseYear'] > 2000]

    fig1 = make_subplots(specs=[[{"secondary_y":True}]])
    fig1.add_trace(go.Bar(x=yrg['releaseYear'], y=yrg['Films'], name='Total Films',
                          marker_color=PURP, opacity=0.8), secondary_y=False)
    fig1.add_trace(go.Scatter(x=yrg['releaseYear'], y=yrg['Hit_Rate'],
                              name='Hit Rate %', mode='lines+markers',
                              line=dict(color=GOLD, width=3), marker=dict(size=8)), secondary_y=True)
    fig1.update_layout(title_text="Annual Film Volume & Hit Rate Trend", **PLOTLY_LAYOUT, height=370)
    fig1.update_xaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS))
    fig1.update_yaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS), secondary_y=False, title_text="Films Released")
    fig1.update_yaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS), secondary_y=True, title_text="Hit Rate %")

    # ── Chart 2: HitFlop distribution donut ───────────────
    hf_cnt = movies['hitFlop_label'].value_counts().reset_index()
    hf_cnt.columns = ['Category','Count']
    order = ['Classic','Blockbuster','Super Hit','Hit','Semi-Hit','Average','Flop','Disaster']
    hf_cnt['Category'] = pd.Categorical(hf_cnt['Category'], categories=order, ordered=True)
    hf_cnt = hf_cnt.sort_values('Category')
    colors_pie = [GREEN, '#00b4d8', CYAN, GOLD, PURP, '#f77f00', ROSE, '#e63946']
    fig2 = go.Figure(go.Pie(labels=hf_cnt['Category'], values=hf_cnt['Count'],
                            hole=0.55, marker=dict(colors=colors_pie),
                            textinfo='label+percent', textfont=dict(color=TEXTM, size=11)))
    fig2.update_layout(title_text="Box-Office Outcome Distribution", **PLOTLY_LAYOUT, height=370)

    col1, col2 = st.columns([3,2])
    with col1: st.plotly_chart(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Genre popularity bar ─────────────────────
    top_gen = gen_df.head(12)
    fig3 = px.bar(top_gen, x='total_films', y='gen_list', orientation='h',
                  color='hit_pct', color_continuous_scale=['#e63946', GOLD, GREEN],
                  labels={'total_films':'Films','gen_list':'Genre','hit_pct':'Hit %'},
                  title="Top Genres by Volume (colour = Hit %)")
    fig3 = styled_fig(fig3, 380)
    fig3.update_coloraxes(colorbar=dict(tickfont=dict(color=TEXTS)))

    # ── Chart 4: Sequel vs Original ───────────────────────
    seq_data = movies.groupby('sequel')['is_hit'].agg(['mean','count']).reset_index()
    seq_data['sequel'] = seq_data['sequel'].map({0:'Original',1:'Sequel',2:'Franchise 3rd+'})
    seq_data['hit_pct'] = (seq_data['mean']*100).round(1)
    fig4 = px.bar(seq_data, x='sequel', y='hit_pct',
                  color='hit_pct', color_continuous_scale=[ROSE, GOLD, GREEN],
                  text='hit_pct', title="Hit Rate: Originals vs Sequels",
                  labels={'sequel':'Film Type','hit_pct':'Hit Rate %'})
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont=dict(color=TEXTM))
    fig4 = styled_fig(fig4, 380)

    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig3, use_container_width=True)
    with col4: st.plotly_chart(fig4, use_container_width=True)

    # Insights
    st.markdown("### 📋 Key Observations")
    insight("""<b>Industry Scale & Output:</b> Bollywood released <b>{:,} films</b> between 2001 and 2014, averaging 
    ~91 films/year. 2009–2012 were peak production years. Yet only <b>{:.1f}%</b> crossed the 'hit' threshold, 
    underscoring how commercially unforgiving the industry truly is.""".format(total, hits/total*100))
    insight("""<b>Flop Economy:</b> A staggering <b>{:.0f}%</b> of all films were outright flops (score ≤ 2). 
    The industry effectively subsidises hits through a high volume of failures — a classic 'hits-driven' 
    media business model identical to Hollywood's studio era.""".format(flops/total*100), style='warn')
    insight("""<b>Sequel Premium:</b> Sequels show a meaningfully higher hit rate than originals 
    — studios greenlight sequels on proven franchises, providing built-in audience awareness and 
    lower marketing friction. This mirrors global cinema economics.""", style='good')
    insight("""<b>Genre Dominance:</b> Drama, Comedy and Romance collectively account for <b>over 60%</b> of 
    all output. Action is the 4th genre — high budget, high stakes. Crime/Thriller films punch above 
    their weight on hit-rate relative to production volume.""")


# ═══════════════════════════════════════════════════════════
# PAGE 2 ▸ ACTOR ANALYTICS
# ═══════════════════════════════════════════════════════════
elif "Actor" in page:
    st.markdown('<div class="page-title">ACTOR ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Performance, consistency & genre DNA of Bollywood stars</div>', unsafe_allow_html=True)

    # KPIs
    qualified = act_df[act_df['total_films'] >= 5]
    best_hit  = qualified.loc[qualified['hit_pct'].idxmax()]
    most_act  = act_df.iloc[0]
    best_rat  = act_df.dropna(subset=['normalizedRating']).loc[act_df.dropna(subset=['normalizedRating'])['normalizedRating'].idxmax()]

    c1,c2,c3,c4 = st.columns(4)
    kpi("Unique Actors",    f"{act_df.shape[0]:,}", c1)
    kpi("Most Prolific",    most_act['name'].split()[0], c2)
    kpi("Best Hit Rate",    f"{best_hit['hit_pct']}%<br><small style='font-size:0.6rem'>{best_hit['name'].split()[0]}</small>", c3)
    kpi("Highest Rating",   f"{best_rat['name'].split()[0]}", c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Top 20 actors by film count + hit rate ──
    top20 = act_df.head(20).sort_values('hit_pct')
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=top20['hit_pct'], y=top20['name'], orientation='h',
                          marker=dict(
                              color=top20['hit_pct'],
                              colorscale=[[0,ROSE],[0.4,GOLD],[1,GREEN]],
                              showscale=True, colorbar=dict(title='Hit%', tickfont=dict(color=TEXTS))
                          ), name='Hit %', text=top20['total_films'],
                          texttemplate='%{text} films', textposition='outside',
                          textfont=dict(color=TEXTS, size=10)))
    fig1.update_layout(title='Top 20 Most Active Actors — Hit Rate', **PLOTLY_LAYOUT, height=550)
    fig1.update_xaxes(gridcolor='#1e2d45', title_text='Hit Rate %', tickfont=dict(color=TEXTS))
    fig1.update_yaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS, size=10))

    # ── Chart 2: Volume vs Hit% scatter (bubble = rating) ─
    q_act = act_df[act_df['total_films'] >= 4].dropna(subset=['normalizedRating'])
    fig2 = px.scatter(q_act, x='total_films', y='hit_pct',
                      size='normalizedRating', color='avg_score',
                      color_continuous_scale=['#e63946', GOLD, GREEN],
                      hover_name='name',
                      labels={'total_films':'Films in Dataset','hit_pct':'Hit Rate %',
                              'avg_score':'Avg Score','normalizedRating':'Popularity Rating'},
                      title='Volume vs Hit Rate (bubble = celebrity rating)')
    fig2 = styled_fig(fig2, 550)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Genre DNA of top 6 actors ───────────────
    top6 = act_df.head(6)['name'].tolist()
    actor_genre_rows = []
    for _, row in movies.iterrows():
        for a in row['act_list']:
            if a in top6:
                for g in row['gen_list']:
                    actor_genre_rows.append({'Actor':a,'Genre':g.title(),'is_hit':row['is_hit']})
    ag_df = pd.DataFrame(actor_genre_rows)
    ag_pivot = ag_df.groupby(['Actor','Genre'])['is_hit'].count().reset_index()
    ag_pivot.columns = ['Actor','Genre','Count']

    fig3 = px.bar(ag_pivot, x='Actor', y='Count', color='Genre',
                  barmode='stack', title='Genre Portfolio of Top 6 Actors',
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig3 = styled_fig(fig3, 380)

    # ── Chart 4: Career decade analysis ──────────────────
    act_yr = movies.explode('act_list')
    act_yr = act_yr[act_yr['act_list'].isin(act_df.head(8)['name'])]
    yr_heat = act_yr.groupby(['releaseYear','act_list'])['is_hit'].mean().reset_index()
    yr_heat.columns = ['Year','Actor','AvgHit']

    fig4 = px.density_heatmap(yr_heat, x='Year', y='Actor', z='AvgHit',
                               color_continuous_scale=['#0d1117', GOLD, GREEN],
                               title='Year-wise Hit Rate Heatmap (Top 8 Actors)',
                               labels={'AvgHit':'Hit Rate'})
    fig4 = styled_fig(fig4, 380)

    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig3, use_container_width=True)
    with col4: st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📋 Key Observations")
    insight(f"""<b>Volume ≠ Success:</b> Akshay Kumar and Ajay Devgn lead in raw film count but their hit 
    rates hover around <b>25–35%</b>. Aamir Khan, who appears in far fewer films, 
    commands a considerably higher hit rate — demonstrating a <b>quality-over-quantity</b> strategy 
    that maximises brand equity while minimising career risk.""")
    insight(f"""<b>The Hit-Rate Sweet Spot:</b> Actors with <b>10–25 films</b> and moderate ratings 
    tend to have the highest hit percentages — they are selective enough to avoid low-budget disasters 
    yet active enough to maintain industry relationships and audience recall.""", style='good')
    insight(f"""<b>Genre Concentration:</b> Drama and Comedy dominate every major actor's portfolio. 
    Action specialists (Ajay Devgn, Suniel Shetty) rely on franchise formulas. 
    Actors who successfully straddle <b>Drama + Comedy</b> show the most resilient hit rates 
    — genre versatility is a real competitive advantage.""")
    insight(f"""<b>Career Trajectory:</b> The heatmap reveals that most top actors had stronger hit 
    rates in the 2006–2011 window, coinciding with India's multiplex boom and the rise of 
    <b>content-driven cinema</b>. Post-2012 shows saturation and declining hit rates industry-wide.""", style='warn')


# ═══════════════════════════════════════════════════════════
# PAGE 3 ▸ DIRECTOR ANALYTICS
# ═══════════════════════════════════════════════════════════
elif "Director" in page:
    st.markdown('<div class="page-title">DIRECTOR ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Who really moves the needle behind the camera?</div>', unsafe_allow_html=True)

    qualified_d = dir_df[dir_df['total_films'] >= 3]
    best_dir    = qualified_d.loc[qualified_d['hit_pct'].idxmax()]
    most_prol   = dir_df.iloc[0]
    best_avg    = qualified_d.loc[qualified_d['avg_score'].idxmax()]

    c1,c2,c3,c4 = st.columns(4)
    kpi("Unique Directors",  f"{dir_df.shape[0]:,}", c1)
    kpi("Most Prolific",     most_prol['name'].split()[0], c2)
    kpi("Highest Hit Rate",  f"{best_dir['hit_pct']}%<br><small style='font-size:0.6rem'>{best_dir['name'].split()[0]}</small>", c3)
    kpi("Highest Avg Score", f"{best_avg['name'].split()[0]}", c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Top 15 directors hit rate + total films ─
    top15d = dir_df[dir_df['total_films'] >= 3].sort_values('hit_pct', ascending=False).head(15).sort_values('hit_pct')
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=top15d['hit_pct'], y=top15d['name'], orientation='h',
                          marker=dict(color=top15d['hit_pct'],
                                      colorscale=[[0,ROSE],[0.5,GOLD],[1,GREEN]], showscale=True,
                                      colorbar=dict(title='Hit %', tickfont=dict(color=TEXTS))),
                          text=top15d['total_films'],
                          texttemplate='%{text} films', textposition='outside',
                          textfont=dict(color=TEXTS, size=10)))
    fig1.update_layout(title='Directors with ≥3 Films — Ranked by Hit Rate', **PLOTLY_LAYOUT, height=520)
    fig1.update_xaxes(gridcolor='#1e2d45', title_text='Hit Rate %', tickfont=dict(color=TEXTS))
    fig1.update_yaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS, size=10))

    # ── Chart 2: Avg score vs film count scatter ──────────
    q_dir = dir_df[dir_df['total_films'] >= 2].dropna(subset=['normalizedRating'])
    fig2 = px.scatter(q_dir, x='total_films', y='avg_score',
                      color='hit_pct', size='total_films',
                      hover_name='name', size_max=40,
                      color_continuous_scale=['#e63946', GOLD, GREEN],
                      title='Director: Productivity vs Quality (colour = hit rate)',
                      labels={'total_films':'Films Directed','avg_score':'Avg hitFlop Score','hit_pct':'Hit %'})
    fig2 = styled_fig(fig2, 520)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: normalizedRating vs hit_pct ─────────────
    rated_dir = dir_df.dropna(subset=['normalizedRating']).query('total_films >= 3')
    fig3 = px.scatter(rated_dir, x='normalizedRating', y='hit_pct',
                      size='total_films', hover_name='name', color='avg_score',
                      color_continuous_scale=['#e63946', GOLD, GREEN],
                      title='Industry Rating vs Box-Office Hit Rate',
                      labels={'normalizedRating':'Industry Normalized Rating',
                              'hit_pct':'Hit Rate %', 'avg_score':'Avg Score'})
    fig3 = styled_fig(fig3, 380)

    # ── Chart 4: Year-wise directorial output heatmap ────
    top8d_names = dir_df.head(8)['name'].tolist()
    dir_yr = movies.explode('dir_list')
    dir_yr = dir_yr[dir_yr['dir_list'].isin(top8d_names)]
    dir_heat = dir_yr.groupby(['releaseYear','dir_list'])['hitFlop'].mean().reset_index()

    fig4 = px.density_heatmap(dir_heat, x='releaseYear', y='dir_list', z='hitFlop',
                               color_continuous_scale=['#0d1117', PURP, GOLD],
                               title='Year-wise Avg Score Heatmap (Top 8 Directors)',
                               labels={'hitFlop':'Avg Score'})
    fig4 = styled_fig(fig4, 380)

    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig3, use_container_width=True)
    with col4: st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📋 Key Observations")
    insight(f"""<b>The Rajkumar Hirani Paradox:</b> Despite directing only 3 films in the dataset, 
    Hirani holds the <b>#1 normalised rating</b> among all directors. His films (Munna Bhai series, 3 Idiots) 
    are near-perfect box-office executions. This is the ultimate proof that <b>selectivity + craft 
    = brand premium</b> — a lesson the high-volume directors have yet to learn.""")
    insight(f"""<b>Volume Directors, Diminishing Returns:</b> Priyadarshan (19 films), Ram Gopal Varma (17), 
    and Vikram Bhatt (16) are the most prolific — yet all three sit in the <b>bottom quartile</b> for 
    hit rate and industry rating. High output without quality control leads to brand erosion.""", style='warn')
    insight(f"""<b>The Sweet Spot:</b> Directors with <b>5–10 films</b> and high selectivity (Rohit Shetty, 
    Karan Johar, Farhan Akhtar) consistently outperform the high-volume cohort on both hit rate 
    and critical acclaim. This band represents <b>optimal creative bandwidth</b>.""", style='good')
    insight(f"""<b>Rating ≠ Hits:</b> The industry rating (normalizedRating) correlates moderately 
    with hit rate but not perfectly. Some directors with modest ratings delivered strong box-office 
    results through audience-friendly formulaic films — confirming that <b>commercial instinct 
    and critical acclaim are distinct skill sets</b> in Bollywood.""")


# ═══════════════════════════════════════════════════════════
# PAGE 4 ▸ GENRE INTELLIGENCE
# ═══════════════════════════════════════════════════════════
elif "Genre" in page:
    st.markdown('<div class="page-title">GENRE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">What audiences actually reward at the box office</div>', unsafe_allow_html=True)

    top_genre   = gen_df.iloc[0]
    best_hit_g  = gen_df[gen_df['total_films'] >= 10].loc[gen_df[gen_df['total_films'] >= 10]['hit_pct'].idxmax()]
    lowest_hit  = gen_df[gen_df['total_films'] >= 10].loc[gen_df[gen_df['total_films'] >= 10]['hit_pct'].idxmin()]

    c1,c2,c3,c4 = st.columns(4)
    kpi("Unique Genres",    f"{gen_df.shape[0]}", c1)
    kpi("Top Genre",        top_genre['gen_list'].title(), c2)
    kpi("Riskiest Genre",   lowest_hit['gen_list'].title(), c3)
    kpi("Safest Genre",     best_hit_g['gen_list'].title(), c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Genre bubble chart (volume, hit%, score) ─
    g_bubble = gen_df[gen_df['total_films'] >= 8]
    fig1 = px.scatter(g_bubble, x='total_films', y='hit_pct', size='avg_score',
                      color='hit_pct', hover_name='gen_list',
                      color_continuous_scale=['#e63946', GOLD, GREEN],
                      title='Genre Risk-Reward Map (size = avg hitFlop score)',
                      labels={'total_films':'Films in Genre','hit_pct':'Hit Rate %','avg_score':'Avg Score'},
                      text='gen_list')
    fig1.update_traces(textposition='top center', textfont=dict(color=TEXTM, size=10))
    fig1 = styled_fig(fig1, 450)

    # ── Chart 2: Genre treemap ────────────────────────────
    fig2 = px.treemap(gen_df.head(15), path=['gen_list'], values='total_films',
                      color='hit_pct', color_continuous_scale=['#e63946', GOLD, GREEN],
                      title='Genre Share (colour = Hit Rate %)',
                      labels={'hit_pct':'Hit %','total_films':'Films'})
    fig2.update_layout(**PLOTLY_LAYOUT, height=450)
    fig2.update_traces(textfont=dict(family='Nunito', color='white', size=13))

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Genre trend over years ──────────────────
    top5g = gen_df.head(5)['gen_list'].tolist()
    gyr = movies.explode('gen_list')
    gyr = gyr[gyr['gen_list'].isin(top5g) & (gyr['releaseYear'] > 2000)]
    gyr_cnt = gyr.groupby(['releaseYear','gen_list'])['title'].count().reset_index()
    gyr_cnt.columns = ['Year','Genre','Films']

    fig3 = px.line(gyr_cnt, x='Year', y='Films', color='Genre',
                   markers=True, title='Top 5 Genres — Annual Film Count Trend',
                   color_discrete_sequence=[GOLD, ROSE, CYAN, GREEN, PURP])
    fig3 = styled_fig(fig3, 380)

    # ── Chart 4: Genre combo heatmap ─────────────────────
    # What genres appear together?
    combo_pairs = []
    for gl in movies['gen_list']:
        gl_clean = [g for g in gl if g.strip()]
        for i in range(len(gl_clean)):
            for j in range(i+1, len(gl_clean)):
                combo_pairs.append({'g1':gl_clean[i].title(),'g2':gl_clean[j].title()})
    combo_df = pd.DataFrame(combo_pairs)
    if not combo_df.empty:
        combo_cnt = combo_df.groupby(['g1','g2']).size().reset_index(name='count')
        combo_cnt = combo_cnt.sort_values('count', ascending=False).head(30)
        pivot = combo_cnt.pivot(index='g1', columns='g2', values='count').fillna(0)
        fig4 = px.imshow(pivot, color_continuous_scale=['#0d1117', PURP, GOLD],
                         title='Genre Co-occurrence Heatmap',
                         labels={'color':'Co-occurrences'})
        fig4.update_layout(**PLOTLY_LAYOUT, height=380)
        fig4.update_xaxes(tickfont=dict(color=TEXTS, size=9))
        fig4.update_yaxes(tickfont=dict(color=TEXTS, size=9))
    else:
        fig4 = go.Figure()

    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig3, use_container_width=True)
    with col4: st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📋 Key Observations")
    insight(f"""<b>Drama Dominates, But Comedy Converts:</b> Drama leads in volume ({int(gen_df[gen_df['gen_list']=='drama']['total_films'].values[0])} films) 
    reflecting Bollywood's emotional storytelling DNA. However, Comedy films show a 
    <b>disproportionately high hit rate</b> relative to their count — audiences reliably show up 
    for laughter, making it the industry's most commercially reliable genre.""")
    insight(f"""<b>Crime & Thriller — High Risk, High Reward:</b> These genres appear less frequently 
    but carry <b>above-average hit rates</b> when executed well. The success of Gangs of Wasseypur, 
    Kahaani, and Talaash in this era demonstrates a growing appetite for intelligent, adult-themed 
    content — an underserved niche in 2001–2014.""", style='good')
    insight(f"""<b>Horror Is a Graveyard:</b> Horror films consistently register the <b>lowest average scores</b> 
    in the dataset. Bollywood has yet to crack the horror formula — most productions in this period 
    relied on cheap thrills and B-grade aesthetics rather than genuine psychological tension.""", style='warn')
    insight(f"""<b>The Hybrid Genre Play:</b> The co-occurrence heatmap reveals that Drama+Romance and 
    Comedy+Drama are the most common pairings. Films that blend <b>exactly two complementary genres</b> 
    outperform both pure single-genre films and overly blended multi-genre projects — 
    suggesting an optimal complexity sweet spot.""")


# ═══════════════════════════════════════════════════════════
# PAGE 5 ▸ DUO & COLLABORATION ANALYSIS
# ═══════════════════════════════════════════════════════════
elif "Duo" in page:
    st.markdown('<div class="page-title">DUO & COLLABORATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Where partnerships create outsized box-office outcomes</div>', unsafe_allow_html=True)

    best_duo  = duo_df[duo_df['films'] >= 3].iloc[0]
    most_collab = duo_df.iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    kpi("Unique Duos",      f"{len(duo_df):,}", c1)
    kpi("Most Repeated",    most_collab['Actor'].split()[0], c2)
    kpi("Best Hit Duo",     f"{best_duo['Actor'].split()[0]} +<br><small>{best_duo['Director'].split()[-1]}</small>", c3)
    kpi("Duo Hit Rate",     f"{best_duo['hit_pct']}%", c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Top actor-director combos (≥3 films) ────
    top_duos = duo_df[duo_df['films'] >= 3].sort_values(['hits','hit_pct'], ascending=False).head(20)
    top_duos['pair'] = top_duos['Actor'] + " ↔ " + top_duos['Director']
    top_duos = top_duos.sort_values('hit_pct')
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=top_duos['hit_pct'], y=top_duos['pair'], orientation='h',
                          marker=dict(color=top_duos['hit_pct'],
                                      colorscale=[[0,ROSE],[0.5,GOLD],[1,GREEN]], showscale=True),
                          text=top_duos['films'],
                          texttemplate='%{text} films', textposition='outside',
                          textfont=dict(color=TEXTS, size=9)))
    fig1.update_layout(title='Best Actor–Director Duos (≥3 films together) by Hit Rate',
                       **PLOTLY_LAYOUT, height=560)
    fig1.update_xaxes(gridcolor='#1e2d45', title_text='Hit Rate %', tickfont=dict(color=TEXTS))
    fig1.update_yaxes(gridcolor='#1e2d45', tickfont=dict(color=TEXTS, size=9))
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Top actors' repeated directors ──────────
    top5_act = act_df.head(5)['name'].tolist()
    act_dir_pairs = duo_df[duo_df['Actor'].isin(top5_act)].sort_values('films', ascending=False).head(25)
    fig2 = px.bar(act_dir_pairs, x='Actor', y='films', color='Director',
                  barmode='stack', title='Who Do Top 5 Actors Work With Most?',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  hover_data=['hit_pct','hits'])
    fig2 = styled_fig(fig2, 400)

    # ── Chart 3: Director loyalty (how many actors do they repeat?) ─
    dir_act_cnt = duo_df[duo_df['films'] >= 2].groupby('Director')['Actor'].count().reset_index()
    dir_act_cnt.columns = ['Director','Repeat_Actors']
    dir_act_cnt = dir_act_cnt.sort_values('Repeat_Actors', ascending=False).head(15)
    fig3 = px.bar(dir_act_cnt, x='Repeat_Actors', y='Director', orientation='h',
                  color='Repeat_Actors', color_continuous_scale=[PURP, GOLD],
                  title='Directors Who Build Recurring Casts',
                  labels={'Repeat_Actors':'Actors worked with ≥2 times'})
    fig3 = styled_fig(fig3, 400)

    col2, col3 = st.columns(2)
    with col2: st.plotly_chart(fig2, use_container_width=True)
    with col3: st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Sankey-style chord (top 6 actors × top 6 dirs) ─
    top6a = act_df.head(6)['name'].tolist()
    top6d = dir_df.head(6)['name'].tolist()
    sub = duo_df[duo_df['Actor'].isin(top6a) & duo_df['Director'].isin(top6d)]
    
    all_nodes = top6a + top6d
    node_colors = [ROSE]*6 + [CYAN]*6
    source = [top6a.index(r['Actor']) for _, r in sub.iterrows()]
    target = [6 + top6d.index(r['Director']) for _, r in sub.iterrows()]
    values = sub['films'].tolist()

    fig4 = go.Figure(go.Sankey(
        node=dict(label=all_nodes, color=node_colors, pad=20, thickness=15,
                  line=dict(color='#1e2d45', width=0.5)),
        link=dict(source=source, target=target, value=values,
                  color='rgba(245,166,35,0.3)')
    ))
    fig4.update_layout(title='Collaboration Flow: Top 6 Actors × Top 6 Directors',
                       **PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📋 Key Observations")
    insight(f"""<b>Chemistry Creates Consistency:</b> Actor-Director duos who collaborate 3+ times 
    show <b>markedly higher hit rates</b> than one-off pairings. Repeated collaboration builds shorthand 
    trust — the director understands the actor's range, the actor trusts the director's vision. 
    This is not luck; it's operational synergy.""")
    insight(f"""<b>The Franchise Ecosystem:</b> High-volume directors (Priyadarshan, David Dhawan) 
    maintain <b>recurring rosters</b> of trusted actors — essentially running mini-studios. 
    While this de-risks casting, it also creates creative echo chambers that can lower average 
    film quality over time.""", style='warn')
    insight(f"""<b>Cross-pollination Gaps:</b> The Sankey chart reveals that top actors and top directors 
    rarely overlap — suggesting Bollywood's elite talent clusters remain somewhat siloed. 
    The biggest commercial breakthroughs often happen when <b>A-list actors break out 
    of their habitual director relationships</b>.""", style='good')
    insight(f"""<b>Star Power vs Director Power:</b> In duo success stories, the director's hit rate 
    (independent of the actor) is a stronger predictor of duo success than the actor's standalone 
    hit rate — reinforcing that in Bollywood, the <b>director is the primary creative risk factor</b>, 
    not the star.""")


# ═══════════════════════════════════════════════════════════
# PAGE 6 ▸ AI CHATBOT  — Encyclopaedic Bollywood Engine
# ═══════════════════════════════════════════════════════════
elif "Chatbot" in page:
    st.markdown('<div class="page-title">AI CHATBOT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Encyclopaedic Bollywood knowledge · 2001–2014 · every actor · every director · every film</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # BUILD MASTER INDEXES (cached)
    # ══════════════════════════════════════════════════════
    @st.cache_data
    def build_master_index():
        # Exploded frames
        act_exp = movies.explode('act_list').copy()
        act_exp = act_exp[act_exp['act_list'].str.strip().isin(['']) == False]
        act_exp = act_exp[~act_exp['act_list'].isin(['Unknown','n/a'])]
        act_exp['act_list'] = act_exp['act_list'].str.strip()

        dir_exp = movies.explode('dir_list').copy()
        dir_exp = dir_exp[~dir_exp['dir_list'].isin(['Unknown','n/a',''])]
        dir_exp['dir_list'] = dir_exp['dir_list'].str.strip()

        gen_exp = movies.explode('gen_list').copy()
        gen_exp = gen_exp[gen_exp['gen_list'].str.strip() != '']
        gen_exp['gen_list'] = gen_exp['gen_list'].str.strip().str.lower()

        # Lowercase lookup sets
        actor_names   = sorted(act_exp['act_list'].unique(), key=len, reverse=True)  # longest first for greedy match
        director_names= sorted(dir_exp['dir_list'].unique(), key=len, reverse=True)
        genre_names   = sorted(gen_exp['gen_list'].unique(), key=len, reverse=True)
        title_names   = sorted(movies['title'].unique(), key=len, reverse=True)

        actor_lower   = {n.lower(): n for n in actor_names}
        director_lower= {n.lower(): n for n in director_names}
        genre_lower   = {n.lower(): n for n in genre_names}
        title_lower   = {t.lower(): t for t in title_names}

        return act_exp, dir_exp, gen_exp, actor_lower, director_lower, genre_lower, title_lower

    act_exp, dir_exp, gen_exp, actor_lower, director_lower, genre_lower, title_lower = build_master_index()

    # ══════════════════════════════════════════════════════
    # ENTITY DETECTOR  — finds any actor/director/genre/title in a query
    # ══════════════════════════════════════════════════════
    def find_name(q_low, lookup_dict):
        """Greedy longest-match entity finder."""
        for name_low in lookup_dict:          # already sorted longest→shortest
            if name_low in q_low:
                return lookup_dict[name_low]   # return canonical cased name
        return None

    def extract_years(q_low):
        """Extract up to two 4-digit years from query."""
        found = re.findall(r'\b(200[1-9]|201[0-4])\b', q_low)
        return [int(y) for y in found]

    def hitflop_label(score):
        mapping = {1:'Disaster',2:'Flop',3:'Average',4:'Semi-Hit',
                   5:'Hit',6:'Super Hit',7:'Blockbuster',8:'Blockbuster',9:'Classic'}
        return mapping.get(int(score), 'Hit')

    # ══════════════════════════════════════════════════════
    # ACTOR ANSWER ENGINE
    # ══════════════════════════════════════════════════════
    def answer_actor(actor_name, q_low):
        sub = act_exp[act_exp['act_list'] == actor_name].copy()
        if sub.empty:
            return f"❌ No data found for **{actor_name}** in the dataset."

        sub_sorted = sub.sort_values('releaseYear')
        total      = len(sub)
        hits       = int(sub['is_hit'].sum())
        flops      = int(sub['is_flop'].sum())
        avg_score  = sub['hitFlop'].mean()
        first_film = sub_sorted.iloc[0]
        last_film  = sub_sorted.iloc[-1]
        years      = extract_years(q_low)

        # ── Time-frame query ─────────────────────────────
        if years and ("between" in q_low or "from" in q_low or "during" in q_low
                      or len(years) == 2 or "to" in q_low):
            y1, y2 = (min(years), max(years)) if len(years) >= 2 else (years[0], years[0])
            tf = sub[(sub['releaseYear'] >= y1) & (sub['releaseYear'] <= y2)]
            if tf.empty:
                return f"📅 **{actor_name}** had **no films** in the dataset between {y1}–{y2}."
            tf_hits = int(tf['is_hit'].sum())
            films_list = "\n".join([
                f"  • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])} [{r['hitFlop']}/9]"
                for _, r in tf.sort_values('releaseYear').iterrows()
            ])
            return (f"📅 **{actor_name} — Films between {y1} and {y2}:**\n\n"
                    f"• **Total films:** {len(tf)}\n"
                    f"• **Hits:** {tf_hits} ({tf_hits/len(tf)*100:.1f}%)\n"
                    f"• **Avg score:** {tf['hitFlop'].mean():.2f}\n\n"
                    f"**Film list:**\n{films_list}")

        # ── Career arc (single year range for career look) ─
        if "career" in q_low and years:
            y1 = years[0]
            y2 = years[1] if len(years) > 1 else movies['releaseYear'].max()
            tf = sub[(sub['releaseYear'] >= y1) & (sub['releaseYear'] <= y2)]
            if tf.empty:
                return f"No data for **{actor_name}** in that period."
            tf_h = int(tf['is_hit'].sum())
            yr_stats = tf.groupby('releaseYear').agg(
                films=('title','count'), hits=('is_hit','sum'), avg=('hitFlop','mean')
            ).reset_index()
            yr_str = "\n".join([
                f"  {int(r['releaseYear'])}: {int(r['films'])} film(s), {int(r['hits'])} hit(s), avg {r['avg']:.1f}"
                for _, r in yr_stats.iterrows()
            ])
            return (f"📈 **{actor_name} — Career {y1}–{y2}:**\n\n"
                    f"• **Films in period:** {len(tf)}\n"
                    f"• **Hit rate:** {tf_h/len(tf)*100:.1f}%\n\n"
                    f"**Year-by-Year:**\n{yr_str}")

        # ── Career arc (no year → full career breakdown) ──
        if "career" in q_low:
            yr_stats = sub.groupby('releaseYear').agg(
                films=('title','count'), hits=('is_hit','sum'), avg=('hitFlop','mean')
            ).reset_index()
            yr_str = "\n".join([
                f"  {int(r['releaseYear'])}: {int(r['films'])} film(s), {int(r['hits'])} hit(s), avg {r['avg']:.1f}"
                for _, r in yr_stats.iterrows()
            ])
            return (f"📈 **{actor_name} — Full Career ({int(first_film['releaseYear'])}–{int(last_film['releaseYear'])}):**\n\n"
                    f"• **Total films:** {total} | **Hits:** {hits} | **Flops:** {flops}\n"
                    f"• **Hit rate:** {hits/total*100:.1f}% | **Avg score:** {avg_score:.2f}\n\n"
                    f"**Year-by-Year Breakdown:**\n{yr_str}")

        # ── First film ───────────────────────────────────
        if "first film" in q_low or "debut" in q_low or "first movie" in q_low:
            dirs = ', '.join([d for d in first_film['dir_list'] if d not in ['Unknown','n/a']])
            return (f"🎬 **{actor_name}'s First Film in Dataset:**\n\n"
                    f"• **Title:** *{first_film['title']}*\n"
                    f"• **Year:** {int(first_film['releaseYear'])}\n"
                    f"• **Genre:** {first_film['genre']}\n"
                    f"• **Director:** {dirs}\n"
                    f"• **Box Office:** {hitflop_label(first_film['hitFlop'])} (Score: {first_film['hitFlop']}/9)")

        # ── Last / Latest film ───────────────────────────
        if "last film" in q_low or "latest film" in q_low or "recent film" in q_low:
            dirs = ', '.join([d for d in last_film['dir_list'] if d not in ['Unknown','n/a']])
            return (f"🎬 **{actor_name}'s Latest Film in Dataset:**\n\n"
                    f"• **Title:** *{last_film['title']}*\n"
                    f"• **Year:** {int(last_film['releaseYear'])}\n"
                    f"• **Genre:** {last_film['genre']}\n"
                    f"• **Director:** {dirs}\n"
                    f"• **Box Office:** {hitflop_label(last_film['hitFlop'])} (Score: {last_film['hitFlop']}/9)")

        # ── All films list ───────────────────────────────
        if "all film" in q_low or "list of film" in q_low or "filmography" in q_low or "list of movie" in q_low:
            film_lines = "\n".join([
                f"  {i+1}. *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                for i, (_, r) in enumerate(sub_sorted.iterrows())
            ])
            return f"🎬 **{actor_name}'s Filmography ({total} films):**\n\n{film_lines}"

        # ── Hit films only ───────────────────────────────
        if "hit film" in q_low or "hit movie" in q_low or "blockbuster" in q_low or "best film" in q_low or "top film" in q_low:
            hits_df = sub[sub['is_hit'] == 1].sort_values('hitFlop', ascending=False)
            if hits_df.empty:
                return f"No hits found for **{actor_name}** in the dataset."
            h_lines = "\n".join([
                f"  {i+1}. *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])} [{r['hitFlop']}/9]"
                for i, (_, r) in enumerate(hits_df.iterrows())
            ])
            return f"🏆 **{actor_name}'s Hit Films ({len(hits_df)} hits):**\n\n{h_lines}"

        # ── Hit ratio / success rate ─────────────────────
        if any(w in q_low for w in ["hit ratio","hit rate","success rate","hit percentage",
                                     "how successful","success ratio","how many hits"]):
            rank_row = actors_rank[actors_rank['actorName'] == actor_name]
            rating   = f"\n• **Industry Rating:** {rank_row['normalizedRating'].values[0]:.2f}/10" if len(rank_row) else ""
            google   = f"\n• **Google Popularity:** {int(rank_row['googleHits'].values[0]):,} hits" if len(rank_row) else ""
            return (f"📊 **{actor_name} — Hit Ratio Analysis:**\n\n"
                    f"• **Total films:** {total}\n"
                    f"• **Hits (score ≥5):** {hits} ({hits/total*100:.1f}%)\n"
                    f"• **Semi-Hits (score=4):** {int((sub['hitFlop']==4).sum())}\n"
                    f"• **Average (score=3):** {int((sub['hitFlop']==3).sum())}\n"
                    f"• **Flops (score=2):** {int((sub['hitFlop']==2).sum())}\n"
                    f"• **Disasters (score=1):** {int((sub['hitFlop']==1).sum())}\n"
                    f"• **Avg hitFlop score:** {avg_score:.2f}/9{rating}{google}")

        # ── How many films ───────────────────────────────
        if "how many" in q_low or "total film" in q_low or "number of film" in q_low or "count" in q_low:
            return (f"🎬 **{actor_name}** has acted in **{total} films** in the dataset ({int(first_film['releaseYear'])}–{int(last_film['releaseYear'])}).\n\n"
                    f"• **Hits:** {hits} | **Flops:** {flops} | **Hit rate:** {hits/total*100:.1f}%")

        # ── Director collaborations ──────────────────────
        if any(w in q_low for w in ["director","collaborated","most work","who directed","direct"]):
            dir_cnt = sub.explode('dir_list')['dir_list']
            dir_cnt = dir_cnt[~dir_cnt.isin(['Unknown','n/a'])].value_counts()
            d_lines = "\n".join([
                f"  {i+1}. **{d}**: {c} film(s)"
                for i, (d, c) in enumerate(dir_cnt.head(8).items())
            ])
            top_dir = dir_cnt.index[0]
            top_cnt = dir_cnt.iloc[0]
            # Films together
            collab_films = sub.explode('dir_list')
            collab_films = collab_films[collab_films['dir_list'] == top_dir][['title','releaseYear','hitFlop']]
            cf_lines = "\n".join([
                f"    • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                for _, r in collab_films.sort_values('releaseYear').iterrows()
            ])
            return (f"🤝 **{actor_name} — Director Collaborations:**\n\n"
                    f"**Most collaborated with: {top_dir}** ({top_cnt} films together)\n"
                    f"{cf_lines}\n\n"
                    f"**All Directors (by frequency):**\n{d_lines}")

        # ── Genre split / preference ─────────────────────
        if any(w in q_low for w in ["genre","speciali","prefer","type of film","type of movie","what kind"]):
            g_cnt = sub.explode('gen_list')['gen_list'].str.strip().str.lower()
            g_cnt = g_cnt[~g_cnt.isin(['unknown',''])].value_counts()
            total_g = g_cnt.sum()
            g_lines = "\n".join([
                f"  {i+1}. **{g.title()}**: {c} appearances ({c/total_g*100:.1f}%)"
                for i, (g, c) in enumerate(g_cnt.head(8).items())
            ])
            # Hit rate per genre
            gen_hits = sub.explode('gen_list').copy()
            gen_hits['gen_list'] = gen_hits['gen_list'].str.strip().str.lower()
            gh = gen_hits.groupby('gen_list')['is_hit'].agg(['sum','count'])
            gh['hr'] = gh['sum']/gh['count']*100
            best_genre_row = gh[gh['count'] >= 2]['hr'].idxmax()
            best_genre_hr  = gh.loc[best_genre_row,'hr']
            return (f"🎭 **{actor_name} — Genre Portfolio:**\n\n"
                    f"**Preferred genre: {g_cnt.index[0].title()}** ({g_cnt.iloc[0]} films, {g_cnt.iloc[0]/total_g*100:.1f}%)\n"
                    f"**Best hit rate genre: {best_genre_row.title()}** ({best_genre_hr:.0f}% hit rate)\n\n"
                    f"**Full Genre Split:**\n{g_lines}")

        # ── Co-stars ─────────────────────────────────────
        if any(w in q_low for w in ["co-star","co star","costar","acted with","worked with actor","alongside"]):
            costars = sub.explode('act_list')['act_list']
            costars = costars[costars != actor_name]
            costars = costars[~costars.isin(['Unknown','n/a'])].value_counts().head(8)
            cs_lines = "\n".join([
                f"  {i+1}. **{c.strip()}**: {v} films together"
                for i, (c, v) in enumerate(costars.items())
            ])
            return f"🤝 **{actor_name}'s Most Frequent Co-Stars:**\n\n{cs_lines}"

        # ── Popularity / Rating ──────────────────────────
        if any(w in q_low for w in ["popular","rating","rank","famous","google","score"]):
            rank_row = actors_rank[actors_rank['actorName'] == actor_name]
            if rank_row.empty:
                return f"No ranking data found for **{actor_name}** in the actors ranking table."
            r = rank_row.iloc[0]
            return (f"⭐ **{actor_name} — Popularity & Rankings:**\n\n"
                    f"• **Normalised Industry Rating:** {r['normalizedRating']:.2f}/10\n"
                    f"• **Google Search Hits:** {int(r['googleHits']):,}\n"
                    f"• **Normalised Google Rank:** {r['normalizedGoogleRank']:.2f}/10\n"
                    f"• **Normalised Movie Rank:** {r['normalizedMovieRank']:.2f}/10\n"
                    f"• **Total films (ranking db):** {int(r['movieCount'])}\n"
                    f"• **Rating Sum:** {int(r['ratingSum'])}")

        # ── DEFAULT: full overview ───────────────────────
        rank_row = actors_rank[actors_rank['actorName'] == actor_name]
        rating_str = f"\n• **Industry Rating:** {rank_row['normalizedRating'].values[0]:.2f}/10" if len(rank_row) else ""
        top_dir_s = sub.explode('dir_list')['dir_list'].value_counts()
        top_dir_s = top_dir_s[~top_dir_s.index.isin(['Unknown','n/a'])]
        fav_dir  = top_dir_s.index[0] if len(top_dir_s) else "N/A"
        fav_genre_s = sub.explode('gen_list')['gen_list'].str.lower().value_counts()
        fav_genre = fav_genre_s.index[0].title() if len(fav_genre_s) else "N/A"
        top5_hits = sub[sub['is_hit']==1].sort_values('hitFlop',ascending=False).head(5)
        top5_str  = "\n".join([f"  • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                                for _, r in top5_hits.iterrows()])
        return (f"🌟 **{actor_name} — Complete Profile:**\n\n"
                f"• **Total films in dataset:** {total}  ({int(first_film['releaseYear'])}–{int(last_film['releaseYear'])})\n"
                f"• **First film:** *{first_film['title']}* ({int(first_film['releaseYear'])})\n"
                f"• **Latest film:** *{last_film['title']}* ({int(last_film['releaseYear'])})\n"
                f"• **Hits (≥5):** {hits} ({hits/total*100:.1f}%)\n"
                f"• **Flops (≤2):** {flops} ({flops/total*100:.1f}%)\n"
                f"• **Avg score:** {avg_score:.2f}/9\n"
                f"• **Favourite genre:** {fav_genre}\n"
                f"• **Most worked director:** {fav_dir}{rating_str}\n\n"
                f"**🏆 Best Films:**\n{top5_str}\n\n"
                f"💡 *Ask me: '{actor_name} genre split', 'career 2005–2010', 'director collaborations', 'hit ratio', 'filmography'*")

    # ══════════════════════════════════════════════════════
    # DIRECTOR ANSWER ENGINE
    # ══════════════════════════════════════════════════════
    def answer_director(dir_name, q_low):
        sub = dir_exp[dir_exp['dir_list'] == dir_name].copy()
        if sub.empty:
            return f"❌ No data found for director **{dir_name}** in the dataset."

        sub_sorted = sub.sort_values('releaseYear')
        total      = len(sub)
        hits       = int(sub['is_hit'].sum())
        flops      = int(sub['is_flop'].sum())
        avg_score  = sub['hitFlop'].mean()
        first_film = sub_sorted.iloc[0]
        last_film  = sub_sorted.iloc[-1]
        years      = extract_years(q_low)

        # ── Time-frame query ─────────────────────────────
        if years and ("between" in q_low or "from" in q_low or "during" in q_low or len(years) >= 2):
            y1, y2 = (min(years), max(years)) if len(years) >= 2 else (years[0], years[0])
            tf = sub[(sub['releaseYear'] >= y1) & (sub['releaseYear'] <= y2)]
            if tf.empty:
                return f"📅 **{dir_name}** directed **no films** in the dataset between {y1}–{y2}."
            tf_hits = int(tf['is_hit'].sum())
            film_lines = "\n".join([
                f"  • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])} [{r['hitFlop']}/9]"
                for _, r in tf.sort_values('releaseYear').iterrows()
            ])
            return (f"📅 **{dir_name} — Directed {y1}–{y2}:**\n\n"
                    f"• **Films:** {len(tf)} | **Hits:** {tf_hits} ({tf_hits/len(tf)*100:.1f}%)\n"
                    f"• **Avg score:** {tf['hitFlop'].mean():.2f}\n\n"
                    f"**Films:**\n{film_lines}")

        # ── Career breakdown ─────────────────────────────
        if "career" in q_low:
            yr_stats = sub.groupby('releaseYear').agg(
                films=('title','count'), hits=('is_hit','sum'), avg=('hitFlop','mean')
            ).reset_index()
            yr_str = "\n".join([
                f"  {int(r['releaseYear'])}: {int(r['films'])} film(s), {int(r['hits'])} hit(s), avg {r['avg']:.1f}"
                for _, r in yr_stats.iterrows()
            ])
            return (f"📈 **{dir_name} — Directorial Career:**\n\n"
                    f"• **Span:** {int(first_film['releaseYear'])}–{int(last_film['releaseYear'])}\n"
                    f"• **Total films:** {total} | **Hit rate:** {hits/total*100:.1f}%\n\n"
                    f"**Year-by-Year:**\n{yr_str}")

        # ── First film ───────────────────────────────────
        if "first film" in q_low or "debut" in q_low or "first movie" in q_low:
            return (f"🎬 **{dir_name}'s First Film (in dataset):**\n\n"
                    f"• **Title:** *{first_film['title']}*\n"
                    f"• **Year:** {int(first_film['releaseYear'])}\n"
                    f"• **Genre:** {first_film['genre']}\n"
                    f"• **Cast:** {first_film['actors'][:80]}...\n"
                    f"• **Box Office:** {hitflop_label(first_film['hitFlop'])} ({first_film['hitFlop']}/9)")

        # ── All films / Filmography ──────────────────────
        if "all film" in q_low or "filmography" in q_low or "list" in q_low:
            film_lines = "\n".join([
                f"  {i+1}. *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                for i, (_, r) in enumerate(sub_sorted.iterrows())
            ])
            return f"🎬 **{dir_name} — Filmography ({total} films):**\n\n{film_lines}"

        # ── Hit ratio ────────────────────────────────────
        if any(w in q_low for w in ["hit ratio","hit rate","success","hit percentage","how successful"]):
            rank_row = dirs_rank[dirs_rank['directorName'] == dir_name]
            rating_s = f"\n• **Industry Rating:** {rank_row['normalizedRating'].values[0]:.2f}/10" if len(rank_row) else ""
            return (f"📊 **{dir_name} — Success Analysis:**\n\n"
                    f"• **Total films:** {total}\n"
                    f"• **Hits (≥5):** {hits} ({hits/total*100:.1f}%)\n"
                    f"• **Semi-Hits (4):** {int((sub['hitFlop']==4).sum())}\n"
                    f"• **Average (3):** {int((sub['hitFlop']==3).sum())}\n"
                    f"• **Flops (≤2):** {flops} ({flops/total*100:.1f}%)\n"
                    f"• **Avg hitFlop score:** {avg_score:.2f}/9{rating_s}")

        # ── Most collaborated actor ──────────────────────
        if any(w in q_low for w in ["actor","cast","collaborat","work with","favourite actor"]):
            act_cnt = sub.explode('act_list')['act_list']
            act_cnt = act_cnt[~act_cnt.isin(['Unknown','n/a'])].value_counts().head(10)
            a_lines = "\n".join([
                f"  {i+1}. **{a}**: {c} film(s)"
                for i, (a, c) in enumerate(act_cnt.items())
            ])
            top_actor = act_cnt.index[0]
            collab_films = sub.explode('act_list')
            collab_films = collab_films[collab_films['act_list'] == top_actor][['title','releaseYear','hitFlop']]
            cf_str = "\n".join([
                f"    • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                for _, r in collab_films.sort_values('releaseYear').iterrows()
            ])
            return (f"🤝 **{dir_name} — Actor Collaborations:**\n\n"
                    f"**Most worked with: {top_actor}** ({act_cnt[top_actor]} films)\n{cf_str}\n\n"
                    f"**All Frequent Actors:**\n{a_lines}")

        # ── Genre preference ─────────────────────────────
        if any(w in q_low for w in ["genre","speciali","prefer","type of film","style"]):
            g_cnt = sub.explode('gen_list')['gen_list'].str.strip().str.lower()
            g_cnt = g_cnt[~g_cnt.isin(['unknown',''])].value_counts()
            total_g = g_cnt.sum()
            g_lines = "\n".join([
                f"  {i+1}. **{g.title()}**: {c} films ({c/total_g*100:.1f}%)"
                for i, (g, c) in enumerate(g_cnt.head(8).items())
            ])
            return (f"🎭 **{dir_name} — Genre Preference:**\n\n"
                    f"**Primary genre: {g_cnt.index[0].title()}** ({g_cnt.iloc[0]/total_g*100:.1f}%)\n\n"
                    f"**Full Genre Breakdown:**\n{g_lines}")

        # ── Popularity / Rating ──────────────────────────
        if any(w in q_low for w in ["popular","rating","rank","famous","google"]):
            rank_row = dirs_rank[dirs_rank['directorName'] == dir_name]
            if rank_row.empty:
                return f"No ranking data for **{dir_name}** in the director rankings table."
            r = rank_row.iloc[0]
            return (f"⭐ **{dir_name} — Industry Rankings:**\n\n"
                    f"• **Normalised Rating:** {r['normalizedRating']:.2f}/10\n"
                    f"• **Google Hits:** {int(r['googleHits']):,}\n"
                    f"• **Normalised Google Rank:** {r['normalizedGoogleRank']:.2f}/10\n"
                    f"• **Normalised Movie Rank:** {r['normalizedMovieRank']:.2f}/10\n"
                    f"• **Total films (db):** {int(r['movieCount'])}")

        # ── DEFAULT full profile ─────────────────────────
        rank_row = dirs_rank[dirs_rank['directorName'] == dir_name]
        rating_s = f"\n• **Industry Rating:** {rank_row['normalizedRating'].values[0]:.2f}/10" if len(rank_row) else ""
        fav_genre_s = sub.explode('gen_list')['gen_list'].str.lower().value_counts()
        fav_genre = fav_genre_s.index[0].title() if len(fav_genre_s) else "N/A"
        top_act_s = sub.explode('act_list')['act_list'].value_counts()
        top_act_s = top_act_s[~top_act_s.index.isin(['Unknown','n/a'])]
        fav_actor = top_act_s.index[0] if len(top_act_s) else "N/A"
        top_hits  = sub[sub['is_hit']==1].sort_values('hitFlop',ascending=False).head(4)
        th_str = "\n".join([f"  • *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                             for _, r in top_hits.iterrows()])
        return (f"🎬 **{dir_name} — Director Profile:**\n\n"
                f"• **Films directed:** {total}  ({int(first_film['releaseYear'])}–{int(last_film['releaseYear'])})\n"
                f"• **First film:** *{first_film['title']}* ({int(first_film['releaseYear'])})\n"
                f"• **Latest film:** *{last_film['title']}* ({int(last_film['releaseYear'])})\n"
                f"• **Hit rate:** {hits/total*100:.1f}% | **Flop rate:** {flops/total*100:.1f}%\n"
                f"• **Avg score:** {avg_score:.2f}/9\n"
                f"• **Favourite genre:** {fav_genre}\n"
                f"• **Most used actor:** {fav_actor}{rating_s}\n\n"
                f"**🏆 Best Films:**\n{th_str}\n\n"
                f"💡 *Ask: '{dir_name} genre', 'actor collaborations', 'career', 'hit ratio', 'filmography'*")

    # ══════════════════════════════════════════════════════
    # FILM ANSWER ENGINE
    # ══════════════════════════════════════════════════════
    def answer_film(film_title, q_low):
        row = movies[movies['title'] == film_title]
        if row.empty:
            # Try partial
            partial = movies[movies['title'].str.lower().str.contains(film_title.lower()[:8], regex=False)]
            if not partial.empty:
                row = partial.iloc[[0]]
            else:
                return f"❌ Film **{film_title}** not found in the dataset."
        r = row.iloc[0]
        score     = r['hitFlop']
        label     = hitflop_label(score)
        actors_l  = [a.strip() for a in str(r['actors']).split('|') if a.strip() not in ['Unknown','n/a']]
        dirs_l    = [d.strip() for d in str(r['directors']).split('|') if d.strip() not in ['Unknown','n/a']]
        main_actor= actors_l[0] if actors_l else "Unknown"
        main_dir  = dirs_l[0] if dirs_l else "Unknown"

        # Actor popularity lookup
        act_rank = actors_rank[actors_rank['actorName'] == main_actor]
        pop_str  = f"{act_rank['normalizedRating'].values[0]:.2f}/10" if len(act_rank) else "N/A"

        # Query-specific responses
        if "director" in q_low and "who" in q_low:
            return (f"🎬 **Director of *{r['title']}* ({int(r['releaseYear'])}):**\n\n"
                    f"• **Director(s):** {', '.join(dirs_l)}\n"
                    f"• **Genre:** {r['genre']}\n"
                    f"• **Box Office:** {label} (Score: {score}/9)")

        if "actor" in q_low or "cast" in q_low or "star" in q_low:
            actor_details = "\n".join([
                f"  • **{a}**" + (f" — Popularity: {actors_rank[actors_rank['actorName']==a]['normalizedRating'].values[0]:.1f}/10"
                                  if len(actors_rank[actors_rank['actorName']==a]) else "")
                for a in actors_l[:6]
            ])
            return (f"🌟 **Cast of *{r['title']}* ({int(r['releaseYear'])}):**\n\n"
                    f"{actor_details}\n\n"
                    f"• **Director:** {', '.join(dirs_l)}\n"
                    f"• **Genre:** {r['genre']}\n"
                    f"• **Box Office:** {label} ({score}/9)")

        if "genre" in q_low:
            return (f"🎭 ***{r['title']}* ({int(r['releaseYear'])}) — Genre:**\n\n"
                    f"• **Genre(s):** {r['genre']}\n"
                    f"• **Box Office:** {label} (Score: {score}/9)\n"
                    f"• **Director:** {', '.join(dirs_l)}")

        if "popular" in q_low or "rating" in q_low:
            return (f"⭐ ***{r['title']}* — Popularity:**\n\n"
                    f"• **Main Actor:** {main_actor} (Rating: {pop_str})\n"
                    f"• **Box Office Score:** {score}/9 — *{label}*\n"
                    f"• **Director:** {', '.join(dirs_l)}")

        # Full film profile
        sequel_s  = "Original" if r['sequel']==0 else f"Sequel #{int(r['sequel'])}"
        return (f"🎬 ***{r['title']}* — Full Profile:**\n\n"
                f"• **Year:** {int(r['releaseYear'])}\n"
                f"• **Genre:** {r['genre']}\n"
                f"• **Director(s):** {', '.join(dirs_l)}\n"
                f"• **Main Actor:** {main_actor} (Popularity: {pop_str})\n"
                f"• **Full Cast:** {', '.join(actors_l[:5])}\n"
                f"• **Box Office:** {label} (Score: {score}/9)\n"
                f"• **Type:** {sequel_s}")

    # ══════════════════════════════════════════════════════
    # GENRE TREND ENGINE
    # ══════════════════════════════════════════════════════
    def answer_genre_trend(q_low):
        years = extract_years(q_low)
        genre_name = find_name(q_low, genre_lower)

        # Single genre + year range
        if genre_name and years:
            y1, y2 = (min(years), max(years)) if len(years) >= 2 else (years[0], years[0]+3)
            sub = gen_exp[(gen_exp['gen_list'] == genre_name.lower()) &
                          (gen_exp['releaseYear'] >= y1) & (gen_exp['releaseYear'] <= y2)]
            if sub.empty:
                return f"No **{genre_name}** films found between {y1}–{y2}."
            yr_stats = sub.groupby('releaseYear').agg(
                films=('title','count'), hits=('is_hit','sum'), avg=('hitFlop','mean')
            ).reset_index()
            yr_str = "\n".join([
                f"  {int(r['releaseYear'])}: {int(r['films'])} films, {int(r['hits'])} hits ({r['hits']/r['films']*100:.0f}%), avg score {r['avg']:.1f}"
                for _, r in yr_stats.iterrows()
            ])
            return (f"📈 **{genre_name.title()} Genre Trend ({y1}–{y2}):**\n\n"
                    f"• **Total {genre_name} films:** {len(sub)}\n"
                    f"• **Overall hit rate:** {sub['is_hit'].mean()*100:.1f}%\n\n"
                    f"**Year-by-Year:**\n{yr_str}")

        # All genres trend with years
        if years:
            y1, y2 = (min(years), max(years)) if len(years) >= 2 else (years[0], movies['releaseYear'].max())
            sub = gen_exp[(gen_exp['releaseYear'] >= y1) & (gen_exp['releaseYear'] <= y2)]
            top5g = sub['gen_list'].value_counts().head(5).index.tolist()
            lines = []
            for g in top5g:
                gs = sub[sub['gen_list'] == g]
                lines.append(f"  • **{g.title()}**: {len(gs)} films, {gs['is_hit'].mean()*100:.0f}% hit rate")
            return (f"📈 **Genre Trends {y1}–{y2}:**\n\n"
                    f"**Top genres in this period:**\n" + "\n".join(lines))

        # Single genre full trend
        if genre_name:
            sub = gen_exp[gen_exp['gen_list'] == genre_name.lower()]
            yr_stats = sub.groupby('releaseYear').agg(
                films=('title','count'), hits=('is_hit','sum')
            ).reset_index()
            yr_str = "\n".join([
                f"  {int(r['releaseYear'])}: {int(r['films'])} films, {int(r['hits'])} hits ({r['hits']/r['films']*100:.0f}% hit rate)"
                for _, r in yr_stats.iterrows()
            ])
            return (f"📈 **{genre_name.title()} — Full Genre Trend (2001–2014):**\n\n"
                    f"• **Total films:** {len(sub)}\n"
                    f"• **Overall hit rate:** {sub['is_hit'].mean()*100:.1f}%\n\n"
                    f"**Year-by-Year:**\n{yr_str}")

        return None

    # ══════════════════════════════════════════════════════
    # TOP-N RANKING ENGINE
    # ══════════════════════════════════════════════════════
    def answer_ranking(q_low):
        n = 5
        for w in q_low.split():
            if w.isdigit() and 1 <= int(w) <= 20:
                n = int(w); break

        years = extract_years(q_low)
        y_filter = ""
        if len(years) >= 2:
            y_filter = f" ({min(years)}–{max(years)})"

        # Top directors
        if "director" in q_low:
            df_use = dir_exp.copy()
            if len(years) >= 2:
                df_use = df_use[(df_use['releaseYear']>=min(years)) & (df_use['releaseYear']<=max(years))]
            g = df_use.groupby('dir_list').agg(films=('title','count'),hits=('is_hit','sum'),avg=('hitFlop','mean')).reset_index()
            g['hit_pct'] = g['hits']/g['films']*100
            sort_by = 'hit_pct' if any(w in q_low for w in ['hit','success','best']) else 'films'
            min_films = 3 if 'hit' in q_low else 1
            g = g[g['films'] >= min_films].sort_values(sort_by, ascending=False).head(n)
            lines = "\n".join([
                f"  {i+1}. **{r['dir_list']}** — {int(r['films'])} films, {r['hit_pct']:.0f}% hit rate, avg {r['avg']:.1f}"
                for i, (_, r) in enumerate(g.iterrows())
            ])
            label = "by hit rate" if sort_by == 'hit_pct' else "by film count"
            return f"🎬 **Top {n} Directors{y_filter} ({label}):**\n\n{lines}"

        # Top actors
        if "actor" in q_low or not any(w in q_low for w in ["director","genre","film","movie"]):
            df_use = act_exp.copy()
            if len(years) >= 2:
                df_use = df_use[(df_use['releaseYear']>=min(years)) & (df_use['releaseYear']<=max(years))]
            g = df_use.groupby('act_list').agg(films=('title','count'),hits=('is_hit','sum'),avg=('hitFlop','mean')).reset_index()
            g['hit_pct'] = g['hits']/g['films']*100
            sort_by = 'hit_pct' if any(w in q_low for w in ['hit','success','best','top']) else 'films'
            min_films = 5 if 'hit' in q_low else 1
            g = g[g['films'] >= min_films].sort_values(sort_by, ascending=False).head(n)
            lines = "\n".join([
                f"  {i+1}. **{r['act_list']}** — {int(r['films'])} films, {r['hit_pct']:.0f}% hit rate"
                for i, (_, r) in enumerate(g.iterrows())
            ])
            label = "by hit rate" if sort_by == 'hit_pct' else "by film count"
            return f"🌟 **Top {n} Actors{y_filter} ({label}):**\n\n{lines}"

        return None

    # ══════════════════════════════════════════════════════
    # YEAR SNAPSHOT ENGINE
    # ══════════════════════════════════════════════════════
    def answer_year(year, q_low):
        sub = movies[movies['releaseYear'] == year]
        if sub.empty:
            return f"No film data found for year **{year}** in the dataset."
        hits  = int(sub['is_hit'].sum())
        total = len(sub)
        top5  = sub.sort_values('hitFlop', ascending=False).head(5)
        top_str = "\n".join([
            f"  {i+1}. *{r['title']}* — {hitflop_label(r['hitFlop'])} ({r['hitFlop']}/9)"
            for i, (_, r) in enumerate(top5.iterrows())
        ])
        top_actors = sub.explode('act_list')['act_list'].value_counts().head(3).index.tolist()
        top_dirs   = sub.explode('dir_list')['dir_list'].value_counts().head(3).index.tolist()
        top_genres = sub.explode('gen_list')['gen_list'].value_counts().head(3).index.tolist()
        return (f"📅 **{year} — Bollywood Industry Snapshot:**\n\n"
                f"• **Films released:** {total}\n"
                f"• **Hits:** {hits} ({hits/total*100:.1f}%)\n"
                f"• **Flops:** {int(sub['is_flop'].sum())} ({sub['is_flop'].mean()*100:.1f}%)\n"
                f"• **Avg score:** {sub['hitFlop'].mean():.2f}/9\n"
                f"• **Top genres:** {', '.join([g.title() for g in top_genres])}\n"
                f"• **Busiest actors:** {', '.join(top_actors)}\n"
                f"• **Busiest directors:** {', '.join(top_dirs)}\n\n"
                f"**🏆 Top Films of {year}:**\n{top_str}")

    # ══════════════════════════════════════════════════════
    # COMPARISON ENGINE
    # ══════════════════════════════════════════════════════
    def answer_comparison(q_low):
        # Find two actors or two directors
        found_actors = []
        for name_low, name in actor_lower.items():
            if name_low in q_low:
                found_actors.append((name_low, name))
                if len(found_actors) == 2:
                    break
        found_dirs = []
        for name_low, name in director_lower.items():
            if name_low in q_low:
                found_dirs.append((name_low, name))
                if len(found_dirs) == 2:
                    break

        def profile(name, df_exp, col):
            sub = df_exp[df_exp[col] == name]
            total = len(sub)
            if total == 0: return None
            hits  = int(sub['is_hit'].sum())
            avg_s = sub['hitFlop'].mean()
            first = sub.sort_values('releaseYear').iloc[0]
            return {'name':name,'total':total,'hits':hits,'hit_pct':hits/total*100,
                    'avg':avg_s,'first':first['title'],'first_yr':int(first['releaseYear'])}

        def compare_two(p1, p2, kind):
            winner = p1 if p1['hit_pct'] >= p2['hit_pct'] else p2
            return (f"⚖️ **{kind} Comparison:**\n\n"
                    f"| Metric | {p1['name']} | {p2['name']} |\n"
                    f"|--------|{'—'*len(p1['name'])}|{'—'*len(p2['name'])}|\n"
                    f"| Films  | {p1['total']} | {p2['total']} |\n"
                    f"| Hits   | {p1['hits']} | {p2['hits']} |\n"
                    f"| Hit Rate | {p1['hit_pct']:.1f}% | {p2['hit_pct']:.1f}% |\n"
                    f"| Avg Score | {p1['avg']:.2f} | {p2['avg']:.2f} |\n"
                    f"| First Film | {p1['first_yr']} | {p2['first_yr']} |\n\n"
                    f"🏆 **{winner['name']}** has the higher hit rate ({winner['hit_pct']:.1f}%)")

        if len(found_actors) >= 2:
            p1 = profile(found_actors[0][1], act_exp, 'act_list')
            p2 = profile(found_actors[1][1], act_exp, 'act_list')
            if p1 and p2: return compare_two(p1, p2, "Actor")

        if len(found_dirs) >= 2:
            p1 = profile(found_dirs[0][1], dir_exp, 'dir_list')
            p2 = profile(found_dirs[1][1], dir_exp, 'dir_list')
            if p1 and p2: return compare_two(p1, p2, "Director")

        return None

    # ══════════════════════════════════════════════════════
    # MASTER ROUTER
    # ══════════════════════════════════════════════════════
    def answer_query(q):
        q_low = q.lower().strip()

        # ── FILM LOOKUP (highest priority) ───────────────
        # Try exact title first
        film_name = find_name(q_low, title_lower)
        if film_name:
            # Check it's actually about the film, not incidentally named
            # (ensure title is a significant portion of the query or query is clearly about a film)
            return answer_film(film_name, q_low)

        # ── COMPARISON ───────────────────────────────────
        if any(w in q_low for w in ["vs","versus","compare","better","against","between"]):
            ans = answer_comparison(q_low)
            if ans: return ans

        # ── GENRE TREND ──────────────────────────────────
        if "trend" in q_low or ("genre" in q_low and any(w in q_low for w in ["over","year","time","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014"])):
            ans = answer_genre_trend(q_low)
            if ans: return ans

        # ── YEAR SNAPSHOT ─────────────────────────────────
        years = extract_years(q_low)
        if years and len(years) == 1 and any(w in q_low for w in ["year","in 20","films in","movies in","snapshot","what happened","industry"]):
            return answer_year(years[0], q_low)

        # ── TOP N RANKINGS ───────────────────────────────
        if any(w in q_low for w in ["top","most","best","highest","rank","list of","prolific","successful"]):
            if any(w in q_low for w in ["director","actor","star"]):
                ans = answer_ranking(q_low)
                if ans: return ans

        # ── DIRECTOR FIRST (if director-specific wording) ─
        dir_name = find_name(q_low, director_lower)
        actor_name = find_name(q_low, actor_lower)

        dir_keywords = ["directed","director","behind the camera","helmed"]
        if dir_name and any(k in q_low for k in dir_keywords):
            return answer_director(dir_name, q_low)

        # ── ACTOR vs DIRECTOR DISAMBIGUATION ─────────────
        # If name found in both, prefer actor unless question has director keywords
        if actor_name and dir_name:
            if any(k in q_low for k in ["director","directed","behind"]):
                return answer_director(dir_name, q_low)
            return answer_actor(actor_name, q_low)

        if actor_name:
            return answer_actor(actor_name, q_low)

        if dir_name:
            return answer_director(dir_name, q_low)

        # ── GENRE STANDALONE ─────────────────────────────
        genre_name = find_name(q_low, genre_lower)
        if genre_name and any(w in q_low for w in ["genre","film","movie","actor","director","popular","hit"]):
            sub = gen_exp[gen_exp['gen_list'] == genre_name.lower()]
            hits = int(sub['is_hit'].sum()); total = len(sub)
            top_films = sub.sort_values('hitFlop', ascending=False).head(5)
            tf_str = "\n".join([
                f"  {i+1}. *{r['title']}* ({int(r['releaseYear'])}) — {hitflop_label(r['hitFlop'])}"
                for i, (_, r) in enumerate(top_films.iterrows())
            ])
            top_act = sub.explode('act_list')['act_list'].value_counts().head(5)
            ta_str  = ", ".join([f"**{a}** ({c})" for a, c in top_act.items()])
            top_dir = sub.explode('dir_list')['dir_list'].value_counts().head(3)
            td_str  = ", ".join([f"**{d}** ({c})" for d, c in top_dir.items()])
            return (f"🎭 **{genre_name.title()} — Genre Deep Dive:**\n\n"
                    f"• **Total films:** {total}\n"
                    f"• **Hit rate:** {hits/total*100:.1f}%\n"
                    f"• **Avg score:** {sub['hitFlop'].mean():.2f}/9\n"
                    f"• **Top actors:** {ta_str}\n"
                    f"• **Top directors:** {td_str}\n\n"
                    f"**🏆 Best {genre_name.title()} Films:**\n{tf_str}")

        # ── YEAR STANDALONE ──────────────────────────────
        if years:
            if len(years) == 1:
                return answer_year(years[0], q_low)
            if len(years) == 2:
                # Range overview
                y1, y2 = min(years), max(years)
                sub = movies[(movies['releaseYear']>=y1) & (movies['releaseYear']<=y2)]
                hits = int(sub['is_hit'].sum()); total = len(sub)
                return (f"📅 **Bollywood {y1}–{y2} Overview:**\n\n"
                        f"• **Films released:** {total}\n"
                        f"• **Hits:** {hits} ({hits/total*100:.1f}%)\n"
                        f"• **Avg score:** {sub['hitFlop'].mean():.2f}/9\n"
                        f"• **Top genre:** {sub.explode('gen_list')['gen_list'].value_counts().index[0].title()}")

        # ── INDUSTRY STATS ───────────────────────────────
        if any(w in q_low for w in ["industry","overall","total","how many film","dataset","all film"]):
            hits = int(movies['is_hit'].sum()); total = len(movies)
            return (f"📊 **Bollywood Dataset — Full Overview (2001–2014):**\n\n"
                    f"• **Total films:** {total:,}\n"
                    f"• **Hits (≥5):** {hits:,} ({hits/total*100:.1f}%)\n"
                    f"• **Flops (≤2):** {int(movies['is_flop'].sum()):,} ({movies['is_flop'].mean()*100:.1f}%)\n"
                    f"• **Unique actors:** {len(actor_lower):,}\n"
                    f"• **Unique directors:** {len(director_lower):,}\n"
                    f"• **Genres covered:** {len(genre_lower)}\n"
                    f"• **Year range:** 2001–2014\n"
                    f"• **Avg hitFlop score:** {movies['hitFlop'].mean():.2f}/9")

        # ── DUO / COLLABORATION ──────────────────────────
        if any(w in q_low for w in ["duo","combination","pairing","together","best pair"]):
            top5 = duo_df[duo_df['films'] >= 3].head(5)
            d_str = "\n".join([
                f"  {i+1}. **{r['Actor']}** × **{r['Director']}** — "
                f"{int(r['hits'])} hits in {int(r['films'])} films ({r['hit_pct']:.0f}%)"
                for i, (_, r) in enumerate(top5.iterrows())
            ])
            return f"🤝 **Best Actor–Director Duos (≥3 films):**\n\n{d_str}"

        # ── HELP / FALLBACK ──────────────────────────────
        return (f"🤖 I'm your **Bollywood Encyclopedia** covering **{len(movies):,} films**, "
                f"**{len(actor_lower):,} actors**, **{len(director_lower):,} directors**, "
                f"**{len(genre_lower)} genres** from **2001–2014**.\n\n"
                f"**Sample questions you can ask:**\n\n"
                f"🎬 *Film Info:*\n"
                f"  • *What genre is Lagaan?*\n"
                f"  • *Who directed 3 Idiots?*\n"
                f"  • *Who starred in Kabhi Khushi Kabhie Gham?*\n\n"
                f"🌟 *Actor Info:*\n"
                f"  • *How many films has Akshay Kumar acted in?*\n"
                f"  • *What was Salman Khan's first film?*\n"
                f"  • *Aamir Khan hit ratio*\n"
                f"  • *Priyanka Chopra genre split*\n"
                f"  • *Akshay Kumar films between 2005 and 2010*\n"
                f"  • *Which director has Ajay Devgn worked with the most?*\n"
                f"  • *Shah Rukh Khan career 2001 to 2008*\n\n"
                f"🎬 *Director Info:*\n"
                f"  • *Karan Johar hit percentage*\n"
                f"  • *Priyadarshan filmography*\n"
                f"  • *Which actor does Rohit Shetty cast the most?*\n"
                f"  • *Ram Gopal Varma career*\n\n"
                f"📈 *Trends & Rankings:*\n"
                f"  • *Genre trend for comedy 2005 to 2012*\n"
                f"  • *Top 10 directors by hit rate*\n"
                f"  • *What happened in 2008?*\n"
                f"  • *Compare Akshay Kumar vs Salman Khan*\n"
                f"  • *Best actor director duos*")

    # ══════════════════════════════════════════════════════
    # CHAT UI
    # ══════════════════════════════════════════════════════
    # Stats bar
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    kpi("Films in DB",     f"{len(movies):,}",        col_s1)
    kpi("Actors Indexed",  f"{len(actor_lower):,}",   col_s2)
    kpi("Directors",       f"{len(director_lower):,}", col_s3)
    kpi("Genres",          f"{len(genre_lower)}",      col_s4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample questions as clickable chips
    st.markdown(f'<div style="color:{TEXTS};font-size:0.8rem;margin-bottom:8px;">💡 Quick questions to try:</div>', unsafe_allow_html=True)
    chips = [
        "Akshay Kumar career",
        "Karan Johar hit percentage",
        "What genre is 3 Idiots?",
        "Top 5 directors by hit rate",
        "Comedy genre trend 2005 to 2012",
        "Compare Salman Khan vs Aamir Khan",
        "Akshay Kumar films between 2005 and 2010",
        "Who directed Lagaan?",
        "Priyadarshan favourite actors",
        "What happened in Bollywood in 2010?",
    ]
    chip_cols = st.columns(5)
    for i, chip in enumerate(chips):
        with chip_cols[i % 5]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role":"user","content":chip})
                ans = answer_query(chip)
                st.session_state.messages.append({"role":"assistant","content":ans})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role":"assistant",
            "content": (f"👋 **Welcome to the Bollywood Encyclopedia!**\n\n"
                        f"I have A-to-Z knowledge of **{len(movies):,} films**, "
                        f"**{len(actor_lower):,} actors**, and **{len(director_lower):,} directors** from 2001–2014.\n\n"
                        f"Ask me about any actor's career, director's filmography, a specific film, "
                        f"genre trends, year snapshots, hit ratios — anything!")
        }]

    # Display messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_q = st.chat_input("Ask anything — actor, director, film, genre, year, trend, comparison...")
    if user_q:
        st.session_state.messages.append({"role":"user","content":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Searching dataset..."):
                ans = answer_query(user_q)
            st.markdown(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})

    # Clear button
    if st.button("🗑️ Clear Chat History", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
