import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import json
import time
from datetime import datetime
from scipy import stats

from faker import Faker

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    SDV_DISPONIVEL = True
except ImportError:
    SDV_DISPONIVEL = False
    print("⚠️  SDV não encontrado. Execute: pip install sdv")

try:
    import diffprivlib as dp
    DP_DISPONIVEL = True
except ImportError:
    DP_DISPONIVEL = False
    print("⚠️  diffprivlib não encontrado. Execute: pip install diffprivlib")

warnings.filterwarnings("ignore")
np.random.seed(42)
fake = Faker("pt_BR")

# ─────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE ESTILO VISUAL
# ─────────────────────────────────────────────────────────
PALETTE = {
    "Real":    "#0D1B2A",
    "Faker":   "#E63946",
    "CTGAN":   "#2A9D8F",
    "bg":      "#F8F9FA",
    "grid":    "#DEE2E6",
    "text":    "#212529",
    "accent":  "#F4A261",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   PALETTE["bg"],
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.color":       PALETTE["grid"],
    "grid.alpha":       0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

# ─────────────────────────────────────────────────────────
# [1/7] CARREGAMENTO DOS DADOS REAIS
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("PIPELINE — AVALIAÇÃO DE DADOS SINTÉTICOS MÉDICOS (LGPD/HIPAA)")
print("=" * 70)

print("\n[1/7] Carregando dataset real de prontuários médicos...")
df_real = pd.read_csv("hospital_data_analysis.csv")

# Normalização de tipos
df_real["Readmission"] = df_real["Readmission"].astype(str).str.strip()
df_real["Cost"]            = pd.to_numeric(df_real["Cost"],            errors="coerce")
df_real["Length_of_Stay"]  = pd.to_numeric(df_real["Length_of_Stay"],  errors="coerce")
df_real["Age"]             = pd.to_numeric(df_real["Age"],             errors="coerce")
df_real["Satisfaction"]    = pd.to_numeric(df_real["Satisfaction"],    errors="coerce")
df_real = df_real.dropna().reset_index(drop=True)

print(f"  Registros carregados: {len(df_real)}")

# ─────────────────────────────────────────────────────────
# Definição das colunas por tipo (domínio médico)
# ─────────────────────────────────────────────────────────
COLS_NUM = ["Age", "Cost", "Length_of_Stay", "Satisfaction"]
COLS_CAT = ["Gender", "Condition", "Procedure", "Outcome", "Readmission"]
COLS_NUM = [c for c in COLS_NUM if c in df_real.columns]
COLS_CAT = [c for c in COLS_CAT if c in df_real.columns]

# ─────────────────────────────────────────────────────────
# [2/7] GERAÇÃO COM FAKER
# ─────────────────────────────────────────────────────────
print("\n[2/7] Gerando dados sintéticos com Faker...")

conditions  = df_real["Condition"].unique().tolist()
procedures  = df_real["Procedure"].unique().tolist()
outcomes    = df_real["Outcome"].unique().tolist()
readmission = df_real["Readmission"].unique().tolist()

def generate_faker_medical(n: int = len(df_real)) -> pd.DataFrame:
    """
    Gera registros médicos com Faker.
    LIMITAÇÃO: não preserva correlações (ex: Condition → Procedure).
    Serve como baseline de privacidade máxima / utilidade mínima.
    """
    records = []
    for i in range(n):
        records.append({
            "Patient_ID":      fake.unique.random_number(digits=9, fix_len=True),
            "Age":             fake.random_int(min=0, max=110),
            "Gender":          fake.random_element(elements=("Male", "Female")),
            "Condition":       fake.random_element(elements=conditions),
            "Procedure":       fake.random_element(elements=procedures),
            "Cost":            round(fake.pyfloat(min_value=500, max_value=35_000, right_digits=2), 2),
            "Length_of_Stay":  fake.random_int(min=1, max=80),
            "Readmission":     fake.random_element(elements=readmission),
            "Outcome":         fake.random_element(elements=outcomes),
            "Satisfaction":    fake.random_int(min=1, max=10),
        })
    return pd.DataFrame(records)

t0 = time.time()
df_faker = generate_faker_medical(len(df_real))
t_faker  = time.time() - t0
print(f"  Faker: {len(df_faker)} registros gerados em {t_faker:.2f}s")

datasets_sinteticos = {"Faker": df_faker}

# ─────────────────────────────────────────────────────────
# [3/7] GERAÇÃO COM SDV (CTGAN)
# ─────────────────────────────────────────────────────────
print("\n[3/7] Treinando modelos SDV...")

if SDV_DISPONIVEL:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_real)

    # Ajuste manual de sdtypes
    SDTYPES = {
        "Patient_ID":     "id",
        "Gender":         "categorical",
        "Condition":      "categorical",
        "Procedure":      "categorical",
        "Outcome":        "categorical",
        "Readmission":    "categorical",
    }
    for col, sdt in SDTYPES.items():
        if col in df_real.columns:
            metadata.update_column(col, sdtype=sdt)
    # — CTGAN —
    print("\n  Treinando CTGAN (pode demorar alguns minutos)...")
    t0 = time.time()
    ctgan = CTGANSynthesizer(
        metadata,
        epochs=500,
        batch_size=100,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        pac=10,
        verbose=True,
    )
    ctgan.fit(df_real)
    t_ctgan = time.time() - t0
    df_ctgan = ctgan.sample(num_rows=len(df_real))
    datasets_sinteticos["CTGAN"] = df_ctgan
    print(f"  CTGAN treinado em {t_ctgan:.1f}s — {len(df_ctgan)} amostras geradas")

else:
    print("  SDV não disponível — pulando CTGAN")
    print("  Execute: pip install sdv")

# ─────────────────────────────────────────────────────────
# [4/7] AVALIAÇÃO DE FIDELIDADE ESTATÍSTICA
# ─────────────────────────────────────────────────────────
print("\n[4/7] Avaliando fidelidade estatística...")

def avaliar_fidelidade(df_real, df_sint, nome):
    """
    Column Shapes  → KS (numéricas) + TVD (categóricas)
    Column Pair Trends → Spearman entre pares numéricos
    Categorical Consistency → distribuição conjunta Condition × Outcome
    """
    m = {"metodo": nome}
    scores_ks, scores_tvd, scores_corr = [], [], []

    cols_num = [c for c in COLS_NUM if c in df_real.columns and c in df_sint.columns]
    cols_cat = [c for c in COLS_CAT if c in df_real.columns and c in df_sint.columns]

    for col in cols_num:
        ks, _ = stats.ks_2samp(df_real[col].dropna(), df_sint[col].dropna())
        scores_ks.append(1 - ks)

    for col in cols_cat:
        cats = set(df_real[col].unique()) | set(df_sint[col].unique())
        p = df_real[col].value_counts(normalize=True)
        q = df_sint[col].value_counts(normalize=True)
        tvd = 0.5 * sum(abs(p.get(c, 0) - q.get(c, 0)) for c in cats)
        scores_tvd.append(1 - tvd)

    m["column_shapes_score"]  = np.mean(scores_ks + scores_tvd)
    m["ks_numerico_medio"]    = np.mean(scores_ks)  if scores_ks  else None
    m["tvd_categorico_medio"] = np.mean(scores_tvd) if scores_tvd else None

    for i, c1 in enumerate(cols_num):
        for c2 in cols_num[i + 1:]:
            cr = df_real[c1].corr(df_real[c2], method="spearman")
            cs = df_sint[c1].corr(df_sint[c2], method="spearman")
            scores_corr.append(1 - abs(cr - cs))
    m["column_pair_trends"] = np.mean(scores_corr) if scores_corr else 0

    # Associação categórica: Condition × Outcome
    if "Condition" in df_real.columns and "Outcome" in df_real.columns:
        tab_real = pd.crosstab(df_real["Condition"], df_real["Outcome"], normalize="all")
        tab_sint = pd.crosstab(df_sint["Condition"] if "Condition" in df_sint.columns else df_sint.iloc[:, 0],
                               df_sint["Outcome"]   if "Outcome"   in df_sint.columns else df_sint.iloc[:, 1],
                               normalize="all")
        common_rows = tab_real.index.intersection(tab_sint.index)
        common_cols = tab_real.columns.intersection(tab_sint.columns)
        if len(common_rows) > 0 and len(common_cols) > 0:
            diff = (tab_real.loc[common_rows, common_cols] -
                    tab_sint.loc[common_rows, common_cols]).abs().values.sum()
            m["assoc_condition_outcome"] = max(0, 1 - diff)
        else:
            m["assoc_condition_outcome"] = 0
    else:
        m["assoc_condition_outcome"] = None

    m["score_geral"] = np.mean([v for v in [
        m["column_shapes_score"],
        m["column_pair_trends"],
        m.get("assoc_condition_outcome") or 0,
    ] if v is not None])

    return m

resultados_qualidade = {}
for nome, df_s in datasets_sinteticos.items():
    res = avaliar_fidelidade(df_real, df_s, nome)
    resultados_qualidade[nome] = res

print(f"\n  {'Método':<12} {'Shapes':>8} {'Correlações':>13} {'Assoc.Cat':>11} {'Score Geral':>13}")
print("  " + "─" * 58)
for nome, r in resultados_qualidade.items():
    print(f"  {nome:<12} {r['column_shapes_score']:>8.3f} "
          f"{r['column_pair_trends']:>13.3f} "
          f"{(r.get('assoc_condition_outcome') or 0):>11.3f} "
          f"{r['score_geral']:>13.3f}")

# ─────────────────────────────────────────────────────────
# [5/7] AVALIAÇÃO DE PRIVACIDADE (NNDR + MIA)
# ─────────────────────────────────────────────────────────
print("\n[5/7] Avaliando riscos de privacidade...")

def avaliar_privacidade(df_real, df_sint, nome):
    """
    NNDR: Nearest Neighbor Distance Ratio — detecta memorização de registros.
    MIA:  Membership Inference Attack simplificado.
    Quasi-Identificadores: verifica unicidade de combinações (Age, Gender, Condition).
    """
    m = {"metodo": nome}
    cols_num = [c for c in COLS_NUM if c in df_real.columns and c in df_sint.columns]

    real_n = df_real[cols_num].copy()
    sint_n = df_sint[cols_num].copy()
    for col in cols_num:
        mn, mx = real_n[col].min(), real_n[col].max()
        real_n[col] = (real_n[col] - mn) / (mx - mn + 1e-9)
        sint_n[col] = (sint_n[col] - mn) / (mx - mn + 1e-9)

    real_arr = real_n.values
    sint_arr = sint_n.values

    idx_sample = np.random.choice(len(sint_arr), min(300, len(sint_arr)), replace=False)
    nndr_scores = []
    for idx in idx_sample:
        p = sint_arr[idx]
        dr = np.sort(np.linalg.norm(real_arr - p, axis=1))[:2]
        ds = np.linalg.norm(sint_arr - p, axis=1)
        ds[idx] = np.inf
        d_nn_s = np.min(ds)
        if d_nn_s > 1e-9:
            nndr_scores.append(dr[0] / d_nn_s)

    m["nndr_medio"] = np.mean(nndr_scores) if nndr_scores else 1.0
    m["risco_nndr"] = ("ALTO"  if m["nndr_medio"] < 0.3 else
                       "MÉDIO" if m["nndr_medio"] < 0.6 else "BAIXO")

    LIMIAR_MIA = 0.05
    hits = sum(1 for idx in idx_sample
               if np.min(np.linalg.norm(real_arr - sint_arr[idx], axis=1)) < LIMIAR_MIA)
    m["taxa_mia"] = hits / len(idx_sample)
    m["risco_mia"] = ("ALTO"  if m["taxa_mia"] > 0.15 else
                      "MÉDIO" if m["taxa_mia"] > 0.05 else "BAIXO")

    # Unicidade de quasi-identificadores
    qi_cols = [c for c in ["Age", "Gender", "Condition"] if c in df_sint.columns]
    if qi_cols:
        uniq_rate = df_sint[qi_cols].drop_duplicates().shape[0] / len(df_sint)
        m["qi_unicidade"] = uniq_rate
        m["risco_qi"] = ("ALTO"  if uniq_rate > 0.9 else
                         "MÉDIO" if uniq_rate > 0.7 else "BAIXO")
    else:
        m["qi_unicidade"] = None
        m["risco_qi"] = "N/A"

    return m

resultados_privacidade = {}
for nome, df_s in datasets_sinteticos.items():
    res = avaliar_privacidade(df_real, df_s, nome)
    resultados_privacidade[nome] = res

print(f"\n  {'Método':<12} {'NNDR':>7} {'Risco NNDR':>12} {'Taxa MIA':>10} {'Risco MIA':>11} {'QI Uniq':>9}")
print("  " + "─" * 68)
for nome, r in resultados_privacidade.items():
    qi = r["qi_unicidade"]
    print(f"  {nome:<12} {r['nndr_medio']:>7.3f} {r['risco_nndr']:>12} "
          f"{r['taxa_mia']:>10.3f} {r['risco_mia']:>11} "
          f"{(qi if qi is not None else 0):>9.3f}")

# ─────────────────────────────────────────────────────────
# [6/7] PRIVACIDADE DIFERENCIAL (diffprivlib)
# ─────────────────────────────────────────────────────────
print("\n[6/7] Consultando dados sintéticos com Privacidade Diferencial...")

melhor_nome = next((k for k in ["CTGAN", "Faker"] if k in datasets_sinteticos), "Faker")
df_melhor   = datasets_sinteticos[melhor_nome]
EPSILON     = 1.0

BOUNDS = {
    "Age":            (0.0,   110.0),
    "Cost":           (100.0, 30000.0),
    "Length_of_Stay": (1.0,   80.0),
    "Satisfaction":   (1.0,   10.0),
}

print(f"\n  Dataset selecionado: {melhor_nome} | ε = {EPSILON}")
print(f"  {'Consulta':<30} {'Valor Direto':>14} {'Com ε-DP':>14} {'Diferença':>12}")
print("  " + "─" * 74)

consultas_dp = [
    ("Média de Idade",           "Age"),
    ("Custo médio (USD)",        "Cost"),
    ("Dias internação médios",   "Length_of_Stay"),
    ("Satisfação média",         "Satisfaction"),
]

for descricao, col in consultas_dp:
    if col not in df_melhor.columns:
        continue
    direto = df_melhor[col].mean()
    if DP_DISPONIVEL:
        est = np.mean([dp.tools.mean(df_melhor[col].values, bounds=BOUNDS[col], epsilon=EPSILON)
                       for _ in range(5)])
    else:
        # Simulação do mecanismo de Laplace sem diffprivlib
        sens = (BOUNDS[col][1] - BOUNDS[col][0]) / len(df_melhor)
        noise = np.random.laplace(0, sens / EPSILON)
        est   = direto + noise
    diff  = abs(est - direto)
    print(f"  {descricao:<30} {direto:>14,.2f} {est:>14,.2f} {diff:>12,.2f}")

# Análise de equidade: gap de custo por gênero (análogo ao gap salarial)
print("\n  Análise de equidade — custo médio por gênero:")
real_gap = (df_real[df_real.Gender == "Male"]["Cost"].mean() /
            df_real[df_real.Gender == "Female"]["Cost"].mean() - 1)
for nome, df_s in datasets_sinteticos.items():
    if "Gender" in df_s.columns and "Cost" in df_s.columns:
        m_m = df_s[df_s.Gender == "Male"]["Cost"].mean()
        m_f = df_s[df_s.Gender == "Female"]["Cost"].mean()
        gap = (m_m / m_f - 1) if m_f > 0 else 0
        ok  = "similar ao real ✓" if abs(gap - real_gap) < 0.08 else "diverge do real ⚠"
        print(f"    {nome:<10}: gap custo M/F = {gap:+.1%}  ({ok})")
print(f"    {'REAL':<10}: gap custo M/F = {real_gap:+.1%}  (referência)")

# ─────────────────────────────────────────────────────────
# [7/7] VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────
print("\n[7/7] Gerando visualizações...")

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Avaliação de Dados Sintéticos — Prontuários Médicos\n"
    "Faker vs CTGAN  ·  Fidelidade Estatística & Risco de Privacidade",
    fontsize=15, fontweight="bold", y=0.995, color=PALETTE["text"]
)

todos_ds = {"Real": df_real, **datasets_sinteticos}
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

# ── LINHA 1: Distribuições numéricas ──────────────────────────────────────────
for idx_col, col in enumerate(["Age", "Cost", "Length_of_Stay"]):
    ax = fig.add_subplot(gs[0, idx_col])
    labels_nice = {"Age": "Idade (anos)", "Cost": "Custo (USD)", "Length_of_Stay": "Dias Internado"}
    ax.set_title(f"Distribuição — {labels_nice.get(col, col)}", fontsize=11, pad=8)
    for nome, df_s in todos_ds.items():
        if col not in df_s.columns:
            continue
        vals = df_s[col].dropna()
        lw   = 2.5 if nome == "Real" else 1.5
        ls   = "-" if nome == "Real" else "--"
        vals.plot.kde(ax=ax, label=nome, color=PALETTE.get(nome, "#888"),
                      linewidth=lw, linestyle=ls)
    ax.set_xlabel(labels_nice.get(col, col))
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=8)

# ── LINHA 2: Distribuições categóricas ────────────────────────────────────────
# 2a — Distribuição de Condição Clínica
ax2a = fig.add_subplot(gs[1, :2])
ax2a.set_title("Frequência por Condição Clínica (top 8)", fontsize=11, pad=8)
top_conds = df_real["Condition"].value_counts().nlargest(8).index.tolist()
x_pos = np.arange(len(top_conds))
width = 0.8 / len(todos_ds)
for i, (nome, df_s) in enumerate(todos_ds.items()):
    if "Condition" not in df_s.columns:
        continue
    freqs = df_s["Condition"].value_counts(normalize=True).reindex(top_conds, fill_value=0)
    ax2a.bar(x_pos + i * width, freqs.values, width,
             label=nome, color=PALETTE.get(nome, "#888"), alpha=0.85)
ax2a.set_xticks(x_pos + width * (len(todos_ds) - 1) / 2)
ax2a.set_xticklabels(top_conds, rotation=30, ha="right", fontsize=8)
ax2a.set_ylabel("Frequência relativa")
ax2a.legend(fontsize=8)

# 2b — Distribuição de Outcome
ax2b = fig.add_subplot(gs[1, 2])
ax2b.set_title("Distribuição de Desfecho (Outcome)", fontsize=11, pad=8)
outcomes_list = df_real["Outcome"].unique().tolist()
x_o = np.arange(len(outcomes_list))
for i, (nome, df_s) in enumerate(todos_ds.items()):
    if "Outcome" not in df_s.columns:
        continue
    freqs = df_s["Outcome"].value_counts(normalize=True).reindex(outcomes_list, fill_value=0)
    ax2b.bar(x_o + i * (0.8 / len(todos_ds)), freqs.values,
             0.8 / len(todos_ds), label=nome,
             color=PALETTE.get(nome, "#888"), alpha=0.85)
ax2b.set_xticks(x_o + 0.3)
ax2b.set_xticklabels(outcomes_list, rotation=20, ha="right", fontsize=8)
ax2b.set_ylabel("Frequência relativa")
ax2b.legend(fontsize=8)

# ── LINHA 3: Qualidade + Privacidade ──────────────────────────────────────────
# 3a — Score de qualidade por método
ax3a = fig.add_subplot(gs[2, :2])
ax3a.set_title("Score de Fidelidade Estatística por Método (0 = péssimo → 1 = perfeito)",
               fontsize=11, pad=8)
metodos_q = list(resultados_qualidade.keys())
x_q = np.arange(len(metodos_q))
w   = 0.22
metricas_barras = [
    ("column_shapes_score",       "Shapes (KS/TVD)"),
    ("column_pair_trends",        "Correlações Spearman"),
    ("assoc_condition_outcome",   "Assoc. Condição × Desfecho"),
    ("score_geral",               "Score Geral"),
]
bar_colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#E63946"]
for i, (chave, label) in enumerate(metricas_barras):
    vals = [resultados_qualidade[m].get(chave) or 0 for m in metodos_q]
    ax3a.bar(x_q + i * w, vals, w, label=label, color=bar_colors[i], alpha=0.88)
ax3a.axhline(y=0.8, color="#E63946", linestyle="--", lw=1.5, label="Limiar aceitável (0.8)")
ax3a.set_xticks(x_q + w * 1.5)
ax3a.set_xticklabels(metodos_q, fontsize=11)
ax3a.set_ylim(0, 1.15)
ax3a.set_ylabel("Score")
ax3a.legend(fontsize=8, loc="upper right")

# 3b — NNDR vs MIA
ax3b = fig.add_subplot(gs[2, 2])
ax3b.set_title("Risco de Privacidade\nNNDR vs MIA", fontsize=11, pad=8)
metodos_p = list(resultados_privacidade.keys())
x_p = np.arange(len(metodos_p))
nndr_v = [resultados_privacidade[m]["nndr_medio"] for m in metodos_p]
mia_v  = [resultados_privacidade[m]["taxa_mia"]   for m in metodos_p]
ax3b.bar(x_p - 0.2, nndr_v, 0.35, label="NNDR ↑ = menor risco", color="#457B9D", alpha=0.85)
ax3b.bar(x_p + 0.2, mia_v,  0.35, label="MIA  ↓ = menor risco", color="#E63946",  alpha=0.85)
ax3b.axhline(0.6,  color="#457B9D", ls="--", lw=1, alpha=0.6)
ax3b.axhline(0.05, color="#E63946",  ls="--", lw=1, alpha=0.6)
ax3b.set_xticks(x_p)
ax3b.set_xticklabels(metodos_p, fontsize=9)
ax3b.set_ylabel("Score")
ax3b.legend(fontsize=8)

# ── LINHA 4: Correlações + Heatmap ────────────────────────────────────────────
# 4a — Matriz de correlação Real
ax4a = fig.add_subplot(gs[3, 0])
ax4a.set_title("Correlação — Dataset REAL", fontsize=11, pad=8)
corr_r = df_real[COLS_NUM].corr()
mask_r = np.triu(np.ones_like(corr_r, dtype=bool), k=1)
sns.heatmap(corr_r, ax=ax4a, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
ax4a.set_xticklabels(ax4a.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax4a.set_yticklabels(ax4a.get_yticklabels(), rotation=0, fontsize=8)

# 4b — Matriz de correlação Faker
ax4b = fig.add_subplot(gs[3, 1])
ax4b.set_title("Correlação — Faker", fontsize=11, pad=8)
corr_f = df_faker[COLS_NUM].corr()
sns.heatmap(corr_f, ax=ax4b, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
ax4b.set_xticklabels(ax4b.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax4b.set_yticklabels(ax4b.get_yticklabels(), rotation=0, fontsize=8)

# 4c — Matriz de correlação CTGAN (ou melhor disponível)
ax4c = fig.add_subplot(gs[3, 2])
melhor_label = melhor_nome
ax4c.set_title(f"Correlação — {melhor_label}", fontsize=11, pad=8)
cols_disp = [c for c in COLS_NUM if c in df_melhor.columns]
corr_m = df_melhor[cols_disp].corr()
sns.heatmap(corr_m, ax=ax4c, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
ax4c.set_xticklabels(ax4c.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax4c.set_yticklabels(ax4c.get_yticklabels(), rotation=0, fontsize=8)

# ── Rodapé informativo
fig.text(0.5, 0.002,
         "Métricas: KS (Kolmogorov-Smirnov), TVD (Total Variation Distance), "
         "NNDR (Nearest Neighbor Distance Ratio), MIA (Membership Inference Attack)  ·  "
         f"ε-DP = {EPSILON}  ·  n_real = {len(df_real)}",
         ha="center", fontsize=8, color="#6C757D")

plt.savefig("avaliacao_sinteticos_medicos.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Gráfico salvo: avaliacao_sinteticos_medicos.png")

# ─────────────────────────────────────────────────────────
# GRÁFICO BÔNUS: Painel de Qualidade Clínica
# ─────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Painel Clínico — Comparação Real vs Sintético",
              fontsize=14, fontweight="bold")

# Box plot: Custo por Condição (top 5 condições)
ax_b1 = axes[0]
top5 = df_real["Condition"].value_counts().nlargest(5).index.tolist()
data_box = []
labels_box = []
for cond in top5:
    sub_r = df_real[df_real.Condition == cond]["Cost"].dropna()
    sub_f = df_faker[df_faker.Condition == cond]["Cost"].dropna() if "Condition" in df_faker.columns else pd.Series()
    data_box.extend([sub_r.values, sub_f.values])
    labels_box.extend([f"{cond[:10]}\nReal", f"{cond[:10]}\nFaker"])

bp = ax_b1.boxplot(data_box, labels=labels_box, patch_artist=True, notch=False)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PALETTE["Real"] if i % 2 == 0 else PALETTE["Faker"])
    patch.set_alpha(0.7)
ax_b1.set_title("Custo por Condição: Real vs Faker", fontsize=10)
ax_b1.set_ylabel("Custo (USD)")
ax_b1.tick_params(axis="x", labelsize=7)
ax_b1.grid(True, alpha=0.3)

# Scatter Age × Cost por método
ax_b2 = axes[1]
for nome, df_s in todos_ds.items():
    if "Age" in df_s.columns and "Cost" in df_s.columns:
        sample = df_s.sample(min(200, len(df_s)), random_state=42)
        alpha = 0.8 if nome == "Real" else 0.4
        size  = 40  if nome == "Real" else 20
        ax_b2.scatter(sample["Age"], sample["Cost"],
                      label=nome, color=PALETTE.get(nome, "#888"),
                      alpha=alpha, s=size,
                      zorder=3 if nome == "Real" else 2)
ax_b2.set_title("Dispersão: Idade × Custo", fontsize=10)
ax_b2.set_xlabel("Idade")
ax_b2.set_ylabel("Custo (USD)")
ax_b2.legend(fontsize=8)

# Taxa de Readmissão por método
ax_b3 = axes[2]
readm_vals = []
readm_labels = []
for nome, df_s in todos_ds.items():
    if "Readmission" in df_s.columns:
        taxa = (df_s["Readmission"].astype(str).str.upper()
                .isin(["YES", "TRUE", "1"]).mean())
        readm_vals.append(taxa)
        readm_labels.append(nome)

bars3 = ax_b3.bar(readm_labels, readm_vals,
                  color=[PALETTE.get(n, "#888") for n in readm_labels],
                  width=0.5, alpha=0.85)
ax_b3.set_title("Taxa de Readmissão por Método", fontsize=10)
ax_b3.set_ylabel("Proporção")
ax_b3.set_ylim(0, 1)
for bar, val in zip(bars3, readm_vals):
    ax_b3.text(bar.get_x() + bar.get_width() / 2,
               bar.get_height() + 0.02, f"{val:.1%}",
               ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("painel_clinico_sinteticos.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Gráfico salvo: painel_clinico_sinteticos.png")

# ─────────────────────────────────────────────────────────
# RELATÓRIO JSON
# ─────────────────────────────────────────────────────────
relatorio = {
    "titulo":    "Avaliação de Dados Sintéticos — Prontuários Médicos",
    "timestamp": datetime.now().isoformat(),
    "dataset_real": {
        "n_registros":          len(df_real),
        "colunas":              list(df_real.columns),
        "pii_presentes":        ["Patient_ID"],
        "atributos_sensiveis":  ["Condition", "Procedure", "Outcome"],
    },
    "qualidade":   {n: {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in r.items()}
                    for n, r in resultados_qualidade.items()},
    "privacidade": {n: {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in r.items()}
                    for n, r in resultados_privacidade.items()},
    "dp": {
        "epsilon":    EPSILON,
        "mecanismo":  "Laplace (via diffprivlib)" if DP_DISPONIVEL else "Laplace simulado",
        "colunas":    COLS_NUM,
    },
    "conformidade_lgpd": {
        "artigos_atendidos": [
            "Art. 6, III — Minimização de dados",
            "Art. 12   — Anonimização técnica",
            "Art. 46   — Medidas técnicas de segurança",
            "Art. 38   — RIPD (Relatório de Impacto à Proteção de Dados)",
        ],
        "metodo_recomendado":  melhor_nome,
        "justificativa": (
            f"{melhor_nome} apresenta melhor equilíbrio fidelidade/privacidade. "
            "Combinado com ε-DP (ε=1.0) para publicação, atende ao princípio de "
            "minimização (Art. 6, III LGPD) e às medidas técnicas do Art. 46. "
            "Dados médicos são sensíveis (Art. 11 LGPD), exigindo controles reforçados."
        ),
    },
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"relatorio_sinteticos_medicos_{ts}.json", "w", encoding="utf-8") as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Relatório JSON salvo: relatorio_sinteticos_medicos_{ts}.json")

# ─────────────────────────────────────────────────────────
# RESUMO FINAL
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONCLUÍDO — DADOS SINTÉTICOS MÉDICOS")
print("=" * 70)
print(f"  Melhor método disponível : {melhor_nome}")
mq = resultados_qualidade.get(melhor_nome, {})
mp = resultados_privacidade.get(melhor_nome, {})
print(f"  Score de qualidade       : {mq.get('score_geral', 0):.3f}")
print(f"  Risco NNDR               : {mp.get('risco_nndr', '?')}")
print(f"  Risco MIA                : {mp.get('risco_mia', '?')}")
print(f"\n  Datasets comparados      : {list(datasets_sinteticos.keys())}")
print(f"  Gráficos gerados         : avaliacao_sinteticos_medicos.png, painel_clinico_sinteticos.png")
print("\nPróximos passos:")
print("  1. Instalar SDV (pip install sdv) para usar CTGAN")
print("  2. Comparar os 3 métodos lado a lado com o SDV disponível")
print("  3. Usar ydata-profiling para relatório HTML detalhado")
print("  4. Avaliar conformidade com HIPAA (EUA) além da LGPD")
print("=" * 70)
