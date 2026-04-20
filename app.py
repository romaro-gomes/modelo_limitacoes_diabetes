import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Preditor de Limitações Funcionais – Diabetes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #f4f6f9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a2638;
    color: #e8edf3;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #b0bfcf !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { display: none; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff;
    font-weight: 600;
}
[data-testid="stSidebar"] .section-divider {
    border-top: 1px solid #2e4060;
    margin: 1rem 0;
}

/* ── Cards ── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 5px solid #2563eb;
}
.metric-card.low    { border-color: #16a34a; }
.metric-card.medium { border-color: #d97706; }
.metric-card.high   { border-color: #dc2626; }

/* ── Gauge number ── */
.gauge-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.2rem;
    font-weight: 500;
    line-height: 1;
}
.gauge-label {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.6rem;
}
.badge-low    { background:#dcfce7; color:#15803d; }
.badge-medium { background:#fef9c3; color:#a16207; }
.badge-high   { background:#fee2e2; color:#b91c1c; }

/* ── Recommendation box ── */
.rec-box {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-top: 0.6rem;
}
.rec-box ul { padding-left: 1.2rem; margin: 0; }
.rec-box li { margin-bottom: 0.4rem; font-size: 0.92rem; color: #374151; }

/* ── Info pills ── */
.pill {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 4px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    modelo = joblib.load('./modelo_treinado_forest.lib')
    processador = joblib.load('./preprocessador_treinado.lib')
    return modelo, processador

modelo, processador = load_models()

# ── Helper: risk level ────────────────────────────────────────────────────────
def risk_level(prob):
    if prob < 0.35:
        return "baixo", "low", "badge-low", "✅ Baixo risco"
    elif prob < 0.60:
        return "moderado", "medium", "badge-medium", "⚠️ Risco moderado"
    else:
        return "alto", "high", "badge-high", "🔴 Alto risco"

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists('./logo_telessaude.jpeg'):
        st.image('./logo_telessaude.jpeg', width=110)
with col_title:
    st.markdown("## Preditor de Limitações Funcionais")
    st.markdown("<span style='color:#64748b;font-size:0.92rem'>Apoio à decisão clínica para pacientes com diabetes · Telessaúde</span>", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR  –  Formulário do paciente
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 👤 Dados do Paciente")

    # ── Perfil ─────────────────────────────────────────────────
    idade = st.slider("Idade (anos)", 18, 120, 55)

    cor_raca = st.selectbox("Cor / Raça", options=[1,2,3,4,5],
        format_func=lambda x: {1:'Branca',2:'Preta',3:'Amarela',4:'Parda',5:'Indígena'}[x])

    renda = st.selectbox("Renda domiciliar per capita", options=[1,2,3,4,5,6,7],
        format_func=lambda x: {
            1:'Até ¼ salário mínimo', 2:'¼ a ½ salário mínimo',
            3:'½ a 1 salário mínimo',  4:'1 a 2 salários mínimos',
            5:'2 a 3 salários mínimos',6:'3 a 5 salários mínimos',
            7:'Mais de 5 salários mínimos'}[x])

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🩺 Histórico Clínico")

    duracao = st.slider("Tempo de diagnóstico (anos)", 0, 50, 5)
    consultas_medico = st.slider("Consultas médicas (últimos 12 meses)", 0, 30, 3)

    insulina = st.radio("Usa insulina prescrita?", options=[1,2],
        format_func=lambda x: {1:"Sim", 2:"Não"}[x], horizontal=True)

    internacao = st.radio("Internação hospitalar ≥ 24h (últimos 12 meses)?",
        options=[1,2], format_func=lambda x: {1:"Sim", 2:"Não"}[x], horizontal=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🥗 Alimentação (dias/semana)")

    c1, c2 = st.columns(2)
    with c1:
        feijao         = st.slider("Feijão",          0, 7, 3)
        carne_vermelha = st.slider("Carne vermelha",  0, 7, 2)
        frango         = st.slider("Frango",          0, 7, 2)
        peixe          = st.slider("Peixe",           0, 7, 1)
        leite          = st.slider("Leite",           0, 7, 3)
    with c2:
        verdura_legume = st.slider("Verdura/Legume",  0, 7, 4)
        frutas         = st.slider("Frutas",          0, 7, 3)
        suco_natural   = st.slider("Suco natural",    0, 7, 2)
        suco_caixinha  = st.slider("Suco industrializado", 0, 7, 1)
        refrigerante   = st.slider("Refrigerante",   0, 7, 1)
        doces          = st.slider("Doces",           0, 7, 1)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📺 Sedentarismo")

    _tv_labels = {1:'< 1h', 2:'1–2h', 3:'2–3h', 4:'3–6h', 5:'≥ 6h', 6:'Não assisto'}
    horas_tv   = st.selectbox("TV por dia", options=[1,2,3,4,5,6],
                              format_func=lambda x: _tv_labels[x])
    horas_tela = st.selectbox("Celular/PC/Tablet (lazer) por dia", options=[1,2,3,4,5,6],
                              format_func=lambda x: _tv_labels[x])

    st.markdown("<br>", unsafe_allow_html=True)
    calcular = st.button("▶  Calcular probabilidade", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════
#  Montar DataFrame
# ═══════════════════════════════════════════════════════════════
data = pd.DataFrame({
    'C008':   [idade],       'C009':   [cor_raca],    'VDF004': [renda],
    'P006':   [feijao],      'P00901': [verdura_legume],'P01101':[carne_vermelha],
    'P013':   [frango],      'P015':   [peixe],       'P01601': [suco_natural],
    'P018':   [frutas],      'P02001': [suco_caixinha],'P02002':[refrigerante],
    'P02501': [doces],       'P023':   [leite],       'P04501': [horas_tv],
    'P04502': [horas_tela],  'Q03802': [insulina],    'J037':   [internacao],
    'J012':   [consultas_medico],                     'duracao':[duracao],
})

# ── Friendly names for explanation chart ─────────────────────
FEATURE_NAMES = {
    'C008':'Idade', 'C009':'Cor/Raça', 'VDF004':'Renda per capita',
    'P006':'Feijão', 'P00901':'Verdura/Legume', 'P01101':'Carne vermelha',
    'P013':'Frango', 'P015':'Peixe', 'P01601':'Suco natural',
    'P018':'Frutas', 'P02001':'Suco industrializado','P02002':'Refrigerante',
    'P02501':'Doces','P023':'Leite','P04501':'Horas de TV',
    'P04502':'Horas de tela (lazer)','Q03802':'Uso de insulina',
    'J037':'Internação hospitalar','J012':'Consultas médicas','duracao':'Tempo de diagnóstico',
}

# ═══════════════════════════════════════════════════════════════
#  MAIN AREA  –  Tabs
# ═══════════════════════════════════════════════════════════════
tab_resultado, tab_explicacao, tab_sobre = st.tabs(
    ["📊 Resultado", "🔍 Entenda a Decisão", "ℹ️ Sobre o Modelo"])

# ────────────────────────────────────────────────────────────────
# TAB 1 – Resultado
# ────────────────────────────────────────────────────────────────
with tab_resultado:
    if not calcular:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#94a3b8'>
            <div style='font-size:3rem'>🩺</div>
            <p style='font-size:1.05rem;margin-top:0.8rem'>
                Preencha o formulário no painel lateral<br>e clique em <strong>Calcular probabilidade</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Calculando..."):
            data_proc = processador.transform(data)
            prob = modelo.predict_proba(data_proc)[0][1]
            st.session_state['prob'] = prob
            st.session_state['data_proc'] = data_proc

        nivel, card_class, badge_class, badge_text = risk_level(prob)
        pct = int(prob * 100)

        # ── Top metric ─────────────────────────────────────────
        st.markdown(f"""
        <div class='metric-card {card_class}'>
            <div class='gauge-label'>Probabilidade de limitação funcional</div>
            <div class='gauge-number'>{pct}%</div>
            <span class='risk-badge {badge_class}'>{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Progress bar visual ────────────────────────────────
        fig_gauge, ax = plt.subplots(figsize=(7, 0.55))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        # background zones
        for zone, color in [(35,'#dcfce7'),(25,'#fef9c3'),(40,'#fee2e2')]:
            pass  # drawn below as a full bar
        ax.barh(0.5, 35,  left=0,  height=0.5, color='#dcfce7', linewidth=0)
        ax.barh(0.5, 25,  left=35, height=0.5, color='#fef9c3', linewidth=0)
        ax.barh(0.5, 40,  left=60, height=0.5, color='#fee2e2', linewidth=0)
        # needle
        ax.axvline(pct, color='#1e3a5f', lw=2.5, ymin=0.1, ymax=0.9)
        ax.scatter([pct], [0.5], color='#1e3a5f', s=80, zorder=5)
        # labels
        for x, lbl in [(17.5,'Baixo'),(47.5,'Moderado'),(80,'Alto')]:
            ax.text(x, 0.5, lbl, ha='center', va='center',
                    fontsize=8, color='#374151', fontweight='500')
        ax.axis('off')
        fig_gauge.patch.set_alpha(0)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # ── Threshold guide ────────────────────────────────────
        st.markdown("""
        <div style='font-size:0.78rem;color:#64748b;margin-top:-0.4rem'>
        <span class='pill'>0–34%</span> Baixo &nbsp;·&nbsp;
        <span class='pill' style='background:#fef9c3;color:#a16207'>35–59%</span> Moderado &nbsp;·&nbsp;
        <span class='pill' style='background:#fee2e2;color:#b91c1c'>≥ 60%</span> Alto
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Clinical recommendations ───────────────────────────
        st.markdown("#### 💡 Orientações Clínicas Sugeridas")

        rec_col1, rec_col2 = st.columns(2)

        if nivel == "baixo":
            with rec_col1:
                st.markdown("""
                <div class='rec-box'>
                <strong>Acompanhamento de rotina</strong>
                <ul>
                  <li>Manter consultas periódicas conforme protocolo</li>
                  <li>Reforçar hábitos saudáveis já estabelecidos</li>
                  <li>Monitorar glicemia e hemoglobina glicada</li>
                </ul>
                </div>""", unsafe_allow_html=True)
            with rec_col2:
                st.markdown("""
                <div class='rec-box'>
                <strong>Prevenção continuada</strong>
                <ul>
                  <li>Incentivar atividade física regular</li>
                  <li>Orientação nutricional básica</li>
                  <li>Reavaliar em 6–12 meses</li>
                </ul>
                </div>""", unsafe_allow_html=True)

        elif nivel == "moderado":
            with rec_col1:
                st.markdown("""
                <div class='rec-box'>
                <strong>⚠️ Intervenção preventiva recomendada</strong>
                <ul>
                  <li>Encaminhar para fisioterapia preventiva</li>
                  <li>Avaliar força muscular e equilíbrio</li>
                  <li>Revisar esquema terapêutico atual</li>
                </ul>
                </div>""", unsafe_allow_html=True)
            with rec_col2:
                st.markdown("""
                <div class='rec-box'>
                <strong>Suporte multidisciplinar</strong>
                <ul>
                  <li>Consulta com nutricionista</li>
                  <li>Programa de atividade física supervisionada</li>
                  <li>Reavaliar em 3 meses</li>
                </ul>
                </div>""", unsafe_allow_html=True)
        else:
            with rec_col1:
                st.markdown("""
                <div class='rec-box'>
                <strong>🔴 Intervenção imediata indicada</strong>
                <ul>
                  <li>Encaminhamento urgente para reabilitação</li>
                  <li>Avaliação geriátrica/funcional completa</li>
                  <li>Revisão completa do plano terapêutico</li>
                </ul>
                </div>""", unsafe_allow_html=True)
            with rec_col2:
                st.markdown("""
                <div class='rec-box'>
                <strong>Suporte intensivo</strong>
                <ul>
                  <li>Equipe multidisciplinar: fisio, nutri, psicologia</li>
                  <li>Monitoramento quinzenal</li>
                  <li>Considerar programa de saúde do idoso</li>
                </ul>
                </div>""", unsafe_allow_html=True)

        # ── Patient summary ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Resumo dos dados inseridos"):
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Idade", f"{idade} anos")
                st.metric("Tempo de diagnóstico", f"{duracao} anos")
                st.metric("Consultas/ano", consultas_medico)
            with s2:
                st.metric("Usa insulina", {1:"Sim",2:"Não"}[insulina])
                st.metric("Internação recente", {1:"Sim",2:"Não"}[internacao])
                st.metric("Renda per capita", {1:'≤ ¼ SM',2:'¼–½ SM',3:'½–1 SM',
                                               4:'1–2 SM',5:'2–3 SM',6:'3–5 SM',7:'> 5 SM'}[renda])
            with s3:
                st.metric("Frutas (dias/sem)", frutas)
                st.metric("Verduras (dias/sem)", verdura_legume)
                st.metric("Refrigerante (dias/sem)", refrigerante)

# ────────────────────────────────────────────────────────────────
# TAB 2 – Entenda a Decisão
# ────────────────────────────────────────────────────────────────
with tab_explicacao:
    if 'data_proc' not in st.session_state:
        st.info("Calcule a probabilidade primeiro (aba Resultado).")
    else:
        st.markdown("#### 🔍 Fatores que mais influenciaram a predição")
        st.markdown(
            "<p style='color:#64748b;font-size:0.88rem'>"
            "O gráfico abaixo mostra a importância relativa de cada variável no modelo treinado "
            "(Random Forest). Fatores com barras maiores têm maior peso nas decisões do modelo "
            "para <em>todos</em> os pacientes — não apenas para este caso individual.</p>",
            unsafe_allow_html=True)

        # Feature importances from the forest
        try:
            importances = modelo.feature_importances_
        except AttributeError:
            importances = np.ones(len(data.columns)) / len(data.columns)

        cols_orig = list(data.columns)
        names_friendly = [FEATURE_NAMES.get(c, c) for c in cols_orig]

        # Sort descending
        idx_sorted = np.argsort(importances)[::-1]
        top_n = min(15, len(idx_sorted))
        top_idx  = idx_sorted[:top_n][::-1]   # reverse so largest is on top in barh
        top_imp  = importances[top_idx]
        top_names = [names_friendly[i] for i in top_idx]

        # Color bars by value quintile
        max_imp = top_imp.max() if top_imp.max() > 0 else 1
        bar_colors = ['#1e40af' if v > 0.6*max_imp
                      else '#3b82f6' if v > 0.3*max_imp
                      else '#93c5fd' for v in top_imp]

        fig, ax = plt.subplots(figsize=(8, top_n * 0.42 + 0.6))
        bars = ax.barh(range(top_n), top_imp, color=bar_colors,
                       edgecolor='white', linewidth=0.5, height=0.65)

        # Value labels
        for bar, val in zip(bars, top_imp):
            ax.text(val + max_imp * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8, color='#374151')

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=9.5)
        ax.set_xlabel("Importância relativa (Gini)", fontsize=9, color='#64748b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlim(0, max_imp * 1.18)
        ax.grid(axis='x', linestyle='--', alpha=0.4, color='#e2e8f0')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── How-to-read legend ─────────────────────────────────
        leg1 = mpatches.Patch(color='#1e40af', label='Alta importância (> 60% do máximo)')
        leg2 = mpatches.Patch(color='#3b82f6', label='Importância intermediária (30–60%)')
        leg3 = mpatches.Patch(color='#93c5fd', label='Baixa importância (< 30%)')

        st.markdown("""
        <div style='font-size:0.8rem;color:#475569;margin-top:0.5rem'>
        <span style='background:#1e40af;padding:2px 8px;border-radius:4px;color:white;margin-right:6px'>■</span>Alta importância &nbsp;
        <span style='background:#3b82f6;padding:2px 8px;border-radius:4px;color:white;margin-right:6px'>■</span>Intermediária &nbsp;
        <span style='background:#93c5fd;padding:2px 8px;border-radius:4px;color:#1e40af;margin-right:6px'>■</span>Baixa importância
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#f1f5f9;border-radius:8px;padding:1rem 1.3rem;font-size:0.85rem;color:#475569'>
        <strong>⚠️ Importante:</strong> Este gráfico reflete a importância global das variáveis no modelo treinado,
        não a contribuição individual para <em>este</em> paciente específico. Para análises individuais (SHAP values),
        certifique-se de que a biblioteca <code>shap</code> está instalada e ative o modo avançado.
        </div>
        """, unsafe_allow_html=True)

        # ── Optionally try SHAP if available ──────────────────
        try:
            import shap
            st.markdown("#### 🧪 Contribuição individual (SHAP values)")
            explainer = shap.TreeExplainer(modelo)
            shap_vals = explainer.shap_values(st.session_state['data_proc'])
            # For binary classification, shap_values returns [class0, class1]
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            sv_names = [FEATURE_NAMES.get(c, c) for c in cols_orig]

            idx_s = np.argsort(np.abs(sv))[::-1][:top_n][::-1]
            shap_sorted = sv[idx_s]
            names_sorted = [sv_names[i] for i in idx_s]
            colors_shap = ['#dc2626' if v > 0 else '#16a34a' for v in shap_sorted]

            fig2, ax2 = plt.subplots(figsize=(8, top_n * 0.42 + 0.6))
            ax2.barh(range(top_n), shap_sorted, color=colors_shap,
                     edgecolor='white', linewidth=0.5, height=0.65)
            ax2.set_yticks(range(top_n))
            ax2.set_yticklabels(names_sorted, fontsize=9.5)
            ax2.set_xlabel("Contribuição SHAP (positivo = aumenta risco)", fontsize=9, color='#64748b')
            ax2.axvline(0, color='#94a3b8', lw=1)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(left=False)
            ax2.grid(axis='x', linestyle='--', alpha=0.4, color='#e2e8f0')
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.markdown("""
            <div style='font-size:0.8rem;color:#475569;margin-top:0.3rem'>
            <span style='background:#dc2626;padding:2px 8px;border-radius:4px;color:white;margin-right:6px'>■</span>Aumenta o risco &nbsp;
            <span style='background:#16a34a;padding:2px 8px;border-radius:4px;color:white;margin-right:6px'>■</span>Reduz o risco
            </div>""", unsafe_allow_html=True)

        except ImportError:
            st.caption("_Instale `shap` para ver contribuições individuais por paciente._")

# ────────────────────────────────────────────────────────────────
# TAB 3 – Sobre
# ────────────────────────────────────────────────────────────────
with tab_sobre:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        ### Sobre o aplicativo
        Este sistema de apoio à decisão clínica foi desenvolvido para auxiliar
        profissionais de saúde a identificar pacientes com diabetes com maior probabilidade
        de desenvolver **limitações funcionais**.

        O modelo preditivo utiliza um **Random Forest** treinado com dados da
        Pesquisa Nacional de Saúde (PNS), considerando variáveis sociodemográficas,
        de alimentação, sedentarismo e histórico clínico.

        ### Como interpretar o resultado
        | Probabilidade | Nível | Conduta sugerida |
        |---|---|---|
        | 0 – 34% | 🟢 Baixo | Acompanhamento de rotina |
        | 35 – 59% | 🟡 Moderado | Iniciar intervenções preventivas |
        | ≥ 60% | 🔴 Alto | Intervenção imediata |

        > O limiar de 35% foi definido com base no equilíbrio entre sensibilidade e especificidade
        > do modelo. Ajuste conforme protocolo institucional.

        ### Limitações
        - O modelo é um **suporte** à decisão, não substitui o julgamento clínico.
        - Treinado em dados populacionais brasileiros — pode não generalizar para todos os contextos.
        - Variáveis de exames laboratoriais não foram incluídas nesta versão.
        """)
    with c2:
        st.markdown("""
        ### Desempenho do modelo
        """)
        metrics_df = pd.DataFrame({
            'Métrica': ['AUC-ROC', 'Sensibilidade', 'Especificidade', 'Acurácia'],
            'Valor': ['—', '—', '—', '—']
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("_Preencha com as métricas do seu modelo treinado._")

        st.markdown("""
        ### Contato
        Para dúvidas técnicas ou sugestões, entre em contato com a equipe de
        desenvolvimento do Telessaúde.
        """)
