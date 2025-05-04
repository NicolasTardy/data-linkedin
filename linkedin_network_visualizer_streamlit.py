"""
Streamlit – Visualiseur LinkedIn (v13 : Skills uploader corrigé)
================================================================
Fonctions :
1. Réseau interactif (Connections.csv).
2. Répertoire contacts par métier + export CSV.
3. Analyse CV vs Offre LinkedIn.
4. Nuage de compétences (Skills.csv uploadé dans la sidebar).
5. Tutoriel détaillé.

Dépendances :
pip install streamlit pandas networkx plotly python-louvain wordcloud python-dotenv requests
# Facultatif :
pip install PyPDF2 reportlab
Exécution :
streamlit run linkedin_network_visualizer_streamlit.py
"""

import os, json, pathlib, requests, re, textwrap
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import community as community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv

# Optionnel : lecture PDF CV
try:
    from PyPDF2 import PdfReader
    CAN_READ_PDF = True
except ImportError:
    CAN_READ_PDF = False

# Optionnel : export PDF des conseils
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    CAN_PDF = True
except ImportError:
    CAN_PDF = False

# ── Charger la clé API ────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="Visualiseur LinkedIn", layout="wide")

# ╔════════ Sidebar – Uploads & paramètres ═════════════════╗
st.sidebar.header("⬆️ Chargez vos données LinkedIn")
con_file       = st.sidebar.file_uploader("Connections.csv", type="csv")
skills_file    = st.sidebar.file_uploader("Skills.csv",      type="csv")
max_company    = st.sidebar.number_input("Top sociétés", 5, 50, 15)
resolution_val = st.sidebar.slider("Clustering réseau", 0.5, 2.0, 1.0, 0.1)

# ╔════════ Helpers ═══════════════════════════════════════╗
CATEGORIES = {
    "Recruteur": ["recruit","talent","headhunter","acquisition"],
    "Directeur": ["director","directeur","head of","chief","vp","vice president"],
    "Manager"  : ["manager","supervisor","responsable"],
    "Fondateur": ["founder","co-founder","cofounder"],
}
def classify_job(title):
    if pd.isna(title): return "Autre"
    low = title.lower()
    for cat,kws in CATEGORIES.items():
        if any(k in low for k in kws): return cat
    return "Autre"

def call_mistral(sys_p, user_p, model="mistral-small-latest", temp=0.6, max_tok=1000):
    if not MISTRAL_KEY:
        st.error("MISTRAL_API_KEY manquante dans .env")
        return ""
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role":"system","content":sys_p}, {"role":"user","content":user_p}],
        "temperature": temp, "max_tokens": max_tok
    }
    r = requests.post("https://api.mistral.ai/v1/chat/completions",
                      headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_json(txt):
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Pas de JSON détecté")
    js = txt[start:end+1]
    js = re.sub(r",\s*([}\]])", r"\1", js)
    return json.loads(js)

# ╔════════ Onglets ═════════════════════════════════════╗
tabs = st.tabs(["🌐 Réseau","📇 Contacts","📄 Analyse CV","☁️ Compétences","ℹ️ Tutoriel"])
net_tab, contact_tab, cv_tab, skill_tab, help_tab = tabs

# ── 1. Réseau ────────────────────────────────────────────
with net_tab:
    st.header("🌐 Carte réseau interactive")
    if not con_file:
        st.info("Chargez Connections.csv")
    else:
        df = pd.read_csv(con_file, skiprows=2, on_bad_lines="skip")
        if df.empty or len(df.columns)<=1:
            con_file.seek(0)
            df = pd.read_csv(con_file, on_bad_lines="skip")
        df["Company"].fillna("Inconnue", inplace=True)
        counts = df["Company"].value_counts()

        st.subheader("Répartition par entreprise")
        fig = go.Figure(go.Bar(
            x=counts.head(max_company).index,
            y=counts.head(max_company).values
        ))
        st.plotly_chart(fig, use_container_width=True)

        G = nx.Graph(); G.add_node("Vous")
        for c,n in counts.items():
            G.add_node(c, size=n); G.add_edge("Vous", c, weight=n)
        part = community_louvain.best_partition(G, resolution=resolution_val)
        cluster = {n:f"Cluster {cid+1}" for n,cid in part.items()}
        groups = sorted({cl for n,cl in cluster.items() if n!="Vous"})
        sel = st.multiselect("Clusters à afficher", groups, default=groups)

        pos = nx.spring_layout(G, seed=42)
        ex,ey,nx_,ny_,ns,nc,nt = [[] for _ in range(7)]
        for u,v in G.edges():
            if v!="Vous" and cluster[v] not in sel: continue
            x0,y0=pos[u]; x1,y1=pos[v]
            ex += [x0,x1,None]; ey += [y0,y1,None]
        pal = px.colors.qualitative.Plotly
        cmap = {cl: pal[i%len(pal)] for i,cl in enumerate(groups)}
        cmap["Vous"] = "#000"
        for n in G.nodes():
            if n!="Vous" and cluster[n] not in sel: continue
            x,y = pos[n]
            nx_.append(x); ny_.append(y)
            ns.append(G.nodes[n].get("size",5)*4)
            nc.append(cmap.get(cluster[n],"#888"))
            nt.append("Vous" if n=="Vous" else f"{n}<br>Connexions: {G.nodes[n]['size']}")
        fig2 = go.Figure([
            go.Scatter(x=ex,y=ey,mode="lines",line=dict(color="#AAA",width=1),hoverinfo="none"),
            go.Scatter(x=nx_,y=ny_,mode="markers+text",text=nt,textposition="bottom center",
                       marker=dict(size=ns,color=nc,line_width=1))
        ])
        fig2.update_layout(height=650,margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig2, use_container_width=True)

# ── 2. Contacts ────────────────────────────────────────────
with contact_tab:
    st.header("📇 Contacts par métier")
    if not con_file:
        st.info("Chargez Connections.csv")
    else:
        con_file.seek(0)
        try:
            dfc = pd.read_csv(con_file, skiprows=2, on_bad_lines="skip")
            if dfc.empty or len(dfc.columns)<=1:
                raise pd.errors.EmptyDataError
        except pd.errors.EmptyDataError:
            con_file.seek(0)
            dfc = pd.read_csv(con_file, on_bad_lines="skip")

        if "Position" not in dfc.columns and "Title" in dfc.columns:
            dfc.rename(columns={"Title":"Position"}, inplace=True)
        dfc["Métier"] = dfc["Position"].apply(classify_job)

        mets = sorted(dfc["Métier"].unique())
        sel  = st.multiselect("Filtrer par métier", mets, default=mets)
        view = dfc[dfc["Métier"].isin(sel)][
            ["First Name","Last Name","Email Address","Company","Position","Métier"]
        ]
        st.dataframe(view, use_container_width=True)
        st.download_button(
            "💾 Export CSV",
            view.to_csv(index=False, encoding="utf-8"),
            file_name="contacts_filtrés.csv",
            mime="text/csv"
        )

# ── 3. Analyse CV ──────────────────────────────────────────
with cv_tab:
    st.header("📄 Analyse CV vs Offre LinkedIn")
    st.markdown(
        "- Uploadez **votre CV PDF** (si PyPDF2 installé) ou collez son texte.\n"
        "- Saisissez l’**URL** de l’offre LinkedIn.\n"
        "- Collez le **texte** de l’offre (titre, description, compétences, expérience).\n"
        "- L’IA propose un **À propos** aligné + **conseils** classés.\n"
        "- Exportez en TXT ou PDF."
    )
    cv_file      = st.file_uploader("CV PDF (facultatif)", type="pdf")
    if cv_file and not CAN_READ_PDF:
        st.warning("pip install PyPDF2 pour lire les PDF")
    cv_text_area = st.text_area("Ou collez le texte de votre CV", height=120)
    job_url      = st.text_input("URL de l’offre LinkedIn")
    offer_text   = st.text_area("Collez le texte complet de l’offre", height=140)

    if st.button("Analyser et conseiller"):
        if not job_url.strip() or not offer_text.strip():
            st.warning("Merci de fournir l’URL et le texte de l’offre.")
            st.stop()

        cv_text = ""
        if cv_file and CAN_READ_PDF:
            reader = PdfReader(cv_file)
            for p in reader.pages:
                cv_text += p.extract_text() or ""
        cv_text = cv_text or cv_text_area

        sys_p = "Vous êtes un coach carrière expert en CV et recrutement."
        user_p = textwrap.dedent(f"""
            Retournez STRICT JSON avec :
            {{
              "about": paragraphe « À propos » (120–150 mots),
              "content": [conseils contenu],
              "structure": [conseils structure],
              "layout": [conseils mise en page]
            }}
            URL offre : {job_url}
            Texte offre : {offer_text}
            Texte CV    : {cv_text}
        """)
        raw = call_mistral(sys_p, user_p, max_tok=1400)
        try:
            result = parse_json(raw)
        except Exception:
            st.error("JSON mal formé, brut IA :"); st.code(raw); st.stop()

        st.markdown("### 📝 À propos proposé")
        st.markdown(result.get("about",""))

        st.markdown("### 💡 Conseils CV")
        for sec,label in [("content","Contenu"),("structure","Structure"),("layout","Mise en page")]:
            items = result.get(sec, [])
            if items:
                st.markdown(f"#### {label}")
                for itm in items:
                    st.markdown(f"- {itm}")

        export_txt = f"URL offre: {job_url}\n\nÀ propos\n{result.get('about','')}\n\n"
        for sec,label in [("content","Contenu"),("structure","Structure"),("layout","Mise en page")]:
            export_txt += f"{label}:\n"
            for itm in result.get(sec,[]): export_txt += f"- {itm}\n"
            export_txt += "\n"
        st.download_button("💾 Télécharger TXT", export_txt.encode("utf-8"),
                           "conseils_cv.txt","text/plain")

        if CAN_PDF:
            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            w,h = A4; y=h-40
            c.setFont("Helvetica-Bold",14); c.drawString(40,y,"Recommandations CV")
            y-=20; c.setFont("Helvetica",10)
            c.drawString(40,y,"URL offre :"); c.drawString(120,y,job_url[:60]+"…")
            y-=25
            c.setFont("Helvetica-Bold",12); c.drawString(40,y,"À propos proposé")
            y-=15; c.setFont("Helvetica",10)
            for ln in textwrap.wrap(result.get("about",""),90):
                c.drawString(40,y,ln); y-=12
            y-=10
            for sec,label in [("content","Contenu"),("structure","Structure"),("layout","Mise en page")]:
                items = result.get(sec,[])
                if items:
                    c.setFont("Helvetica-Bold",12); c.drawString(40,y,label); y-=15
                    c.setFont("Helvetica",10)
                    for itm in items:
                        for ln in textwrap.wrap(itm,90):
                            c.drawString(50,y,ln); y-=12
                        y-=6
                        if y<80:
                            c.showPage(); y=h-40; c.setFont("Helvetica",10)
            c.save(); buf.seek(0)
            st.download_button("📄 Télécharger PDF", buf,
                               "conseils_cv.pdf","application/pdf")
        else:
            st.warning("pip install reportlab pour exporter en PDF")

# ── 4. Nuage de compétences ─────────────────────────────────
with skill_tab:
    st.header("☁️ Nuage de compétences")
    if not skills_file:
        st.info("Chargez Skills.csv dans la barre latérale")
    else:
        df_sk = pd.read_csv(skills_file, on_bad_lines="skip")
        names = df_sk["Name"].dropna().tolist()
        if not names:
            st.info("Aucune compétence détectée")
        else:
            wc = WordCloud(width=1000,height=500,background_color="white")\
                 .generate(" ".join(names))
            fig,ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig)
            buf = BytesIO(); fig.savefig(buf, format="png")
            st.download_button("💾 Export PNG", buf.getvalue(),
                               "nuage_competences.png","image/png")

# ── 5. Tutoriel ─────────────────────────────────────────────
with help_tab:
    st.header("ℹ️ Tutoriel détaillé & Carte")
    st.markdown(
        "### 1. Export LinkedIn\n"
        "Vous → Préférences & confidentialité → Confidentialité des données → Obtenir une copie → Données complètes → ZIP → dézippez.\n\n"
        "### 2. Carte réseau\n"
        "- Nœud = entreprise, centre = vous.\n"
        "- Taille/épaisseur ∝ connexions.\n"
        "- Clustering Louvain regroupe secteurs.\n"
        "- Filtrez par cluster pour isoler un segment.\n\n"
        "### 3. Analyse CV vs Offre\n"
        "- Uploadez ou collez votre CV + saisissez l’URL + collez le texte de l’offre.\n"
        "- L’IA génère un À propos aligné et des conseils classés.\n"
        "- Exportez en TXT ou PDF.\n\n"
        "### 4. Nuage compétences\n"
        "- Chargez Skills.csv dans la barre latérale → wordcloud exportable."
    )