"""
Interface web em Gradio para classificação de risco de lesão.

Fornece um formulário interativo para atletas inserirem dados de saúde
e desempenho e receberem previsões de risco de lesão em tempo real.
"""

from pathlib import Path
import re
import sys

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import load_model
from risk_score import compute_risk_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_message(message: str) -> str:
    """Hide absolute project paths in logs and UI-facing error messages."""
    sanitized = str(message).replace(str(PROJECT_ROOT), ".")
    sanitized = sanitized.replace(f"file://{PROJECT_ROOT}", ".")
    sanitized = re.sub(r"/home/[^\s:'\")]+", "[absolute-path]", sanitized)
    return sanitized


# Carrega o modelo uma unica vez na inicializacao
try:
    models_dir = Path(__file__).parent.parent / "models"
    predictor = load_model(models_dir)
    model_loaded = True
except Exception as e:
    print(f"Aviso: nao foi possivel carregar o modelo: {_sanitize_message(str(e))}")
    model_loaded = False
    predictor = None


RISK_SIGNAL_SPECS = [
    ("post_discomfort", "Desconforto pos-atividade", 0.0, 1.0, False),
    ("rpe", "Alta exigencia (RPE)", 1.0, 10.0, False),
    ("fatigue_level", "Nivel de fadiga", 1.0, 10.0, False),
    ("sleep_hours", "Poucas horas de sono", 0.0, 10.0, True),
    ("sleep_quality", "Baixa qualidade do sono", 1.0, 5.0, True),
    ("stress_level", "Alto estresse", 1.0, 3.0, False),
]

RADAR_SIGNAL_SPECS = RISK_SIGNAL_SPECS

HEURISTIC_FACTOR_SPECS = [
    ("post_discomfort", "Desconforto pos", 2.0, lambda row: row.get("post_discomfort") == 1),
    ("rpe_high", "RPE >= 8", 1.0, lambda row: pd.notna(row.get("rpe")) and row.get("rpe") >= 8),
    ("fatigue_high", "Fadiga >= 7", 1.0, lambda row: pd.notna(row.get("fatigue_level")) and row.get("fatigue_level") >= 7),
    ("pre_discomfort", "Desconforto pre", 1.0, lambda row: row.get("pre_discomfort") == 1),
    ("sleep_insufficient", "Sono < 6h", 1.0, lambda row: pd.notna(row.get("sleep_hours")) and row.get("sleep_hours") < 6),
    ("stress_high", "Estresse alto", 1.0, lambda row: row.get("stress_level") == 3),
    ("sleep_poor_quality", "Sono de baixa qualidade", 1.0, lambda row: row.get("sleep_quality") == 1),
]


def _normalize_risk_signal(series: pd.Series, minimum: float, maximum: float, invert: bool) -> pd.Series:
    span = maximum - minimum
    if span <= 0:
        return pd.Series(0.0, index=series.index)

    normalized = (pd.to_numeric(series, errors="coerce") - minimum) / span
    normalized = normalized.clip(0.0, 1.0).fillna(0.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def _build_explanation_frame(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or 'nickname' not in results.columns:
        return pd.DataFrame()

    frame = pd.DataFrame(index=results.index)

    if 'position' in results.columns:
        frame['player_label'] = results['nickname'].astype(str) + ' (' + results['position'].astype(str) + ')'
    else:
        frame['player_label'] = results['nickname'].astype(str)

    if 'risk_score' in results.columns:
        frame['risk_score'] = pd.to_numeric(results['risk_score'], errors='coerce').fillna(0.0)
    else:
        frame['risk_score'] = results.apply(compute_risk_score, axis=1)

    if 'prob_high' in results.columns:
        frame['prob_high'] = pd.to_numeric(results['prob_high'], errors='coerce').fillna(0.0)
    if 'confidence' in results.columns:
        frame['confidence'] = pd.to_numeric(results['confidence'], errors='coerce').fillna(0.0)
    if 'risk_label' in results.columns:
        frame['risk_label'] = results['risk_label'].astype(str)

    for column_name, display_name, minimum, maximum, invert in RISK_SIGNAL_SPECS:
        if column_name in results.columns:
            frame[display_name] = _normalize_risk_signal(results[column_name], minimum, maximum, invert)

    frame = frame.sort_values(['risk_score', 'prob_high'] if 'prob_high' in frame.columns else ['risk_score'], ascending=False)
    return frame.reset_index(drop=True)


def _build_heuristic_breakdown_frame(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or 'nickname' not in results.columns:
        return pd.DataFrame()

    frame = pd.DataFrame(index=results.index)
    if 'position' in results.columns:
        frame['player_label'] = results['nickname'].astype(str) + ' (' + results['position'].astype(str) + ')'
    else:
        frame['player_label'] = results['nickname'].astype(str)

    contributions = []
    for _, row in results.iterrows():
        row_contributions = {}
        for _, display_name, weight, condition in HEURISTIC_FACTOR_SPECS:
            row_contributions[display_name] = weight if condition(row) else 0.0
        contributions.append(row_contributions)

    contributions_df = pd.DataFrame(contributions, index=results.index)
    frame = pd.concat([frame, contributions_df], axis=1)
    frame['risk_score'] = frame[[spec[1] for spec in HEURISTIC_FACTOR_SPECS]].sum(axis=1)
    if 'risk_score' in results.columns:
        frame['risk_score'] = pd.to_numeric(results['risk_score'], errors='coerce').fillna(frame['risk_score'])
    if 'prob_high' in results.columns:
        frame['prob_high'] = pd.to_numeric(results['prob_high'], errors='coerce').fillna(0.0)
    return frame.sort_values(['risk_score', 'prob_high'] if 'prob_high' in frame.columns else ['risk_score'], ascending=False).reset_index(drop=True)


def build_risk_signal_heatmap(results: pd.DataFrame):
    """Constroi um mapa de calor com sinais de risco normalizados por atleta."""
    plot_df = _build_explanation_frame(results)
    signal_columns = [display_name for _, display_name, _, _, _ in RISK_SIGNAL_SPECS if display_name in plot_df.columns]

    if plot_df.empty or not signal_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado explicativo disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    heatmap_df = plot_df.loc[:, signal_columns].copy()
    fig_height = max(4, len(heatmap_df) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    im = ax.imshow(heatmap_df.values, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(plot_df['player_label'])
    ax.set_title('Mapa de calor de sinais de risco por atleta', fontweight='bold')
    ax.set_xlabel('Sinais normalizados (maior = mais preocupante)')
    ax.set_ylabel('Atleta')

    for i in range(len(heatmap_df)):
        for j in range(len(heatmap_df.columns)):
            ax.text(j, i, f'{heatmap_df.values[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=ax, label='Sinal de risco normalizado')
    plt.tight_layout()
    return fig


def build_heuristic_breakdown_plot(results: pd.DataFrame):
    """Constroi um grafico de barras empilhadas com o score heuristico por atleta."""
    plot_df = _build_heuristic_breakdown_frame(results)
    factor_columns = [display_name for _, display_name, _, _ in HEURISTIC_FACTOR_SPECS]

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum detalhamento de score disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    fig_height = max(6.8, min(8.0, len(plot_df) * 0.62))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = ['#C0392B', '#E67E22', '#F1C40F', '#3498DB', '#9B59B6', '#16A085', '#7F8C8D']
    left = np.zeros(len(plot_df))
    for index, factor_name in enumerate(factor_columns):
        values = plot_df[factor_name].to_numpy(dtype=float)
        ax.barh(plot_df['player_label'], values, left=left, color=colors[index % len(colors)], label=factor_name)
        left = left + values

    for y_index, score in enumerate(plot_df['risk_score']):
        ax.text(score + 0.05, y_index, f'{score:.0f}', va='center', fontweight='bold')

    ax.set_title('Detalhamento do Score Heuristico por Atleta', fontweight='bold')
    ax.set_xlabel('Contribuicao para o score de risco')
    ax.set_ylabel('Atleta')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def build_top_risk_radar(results: pd.DataFrame):
    """Constroi um grafico radar para os atletas com maior risco previsto."""
    plot_df = _build_explanation_frame(results)
    radar_columns = [display_name for _, display_name, _, _, _ in RADAR_SIGNAL_SPECS if display_name in plot_df.columns]

    if plot_df.empty or not radar_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado para radar disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    top_df = plot_df.head(min(3, len(plot_df))).copy()
    values = top_df.loc[:, radar_columns].to_numpy(dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(radar_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    color_map = ['#C0392B', '#E69F00', '#2E8B57']

    for index, (_, row) in enumerate(top_df.iterrows()):
        row_values = values[index].tolist()
        row_values += row_values[:1]
        ax.plot(angles, row_values, color=color_map[index % len(color_map)], linewidth=2, label=row['player_label'])
        ax.fill(angles, row_values, color=color_map[index % len(color_map)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_columns)
    ax.set_yticklabels([])
    ax.set_title('Radar dos perfis com maior risco', fontweight='bold', pad=20)
    ax.text(
        0.5,
        1.08,
        'Todos os eixos sao normalizados em 0-1. Valores maiores indicam sinal de risco mais forte.',
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=9,
    )
    ax.legend(title='Top atletas por risco', loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    return fig


def build_top_players_vs_average_plot(results: pd.DataFrame):
    """Constroi barras agrupadas comparando top atletas com a media do lote."""
    plot_df = _build_explanation_frame(results)
    compare_columns = [display_name for _, display_name, _, _, _ in RADAR_SIGNAL_SPECS if display_name in plot_df.columns]

    if plot_df.empty or not compare_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de comparacao disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    top_df = plot_df.head(min(3, len(plot_df))).copy()
    average_values = plot_df.loc[:, compare_columns].mean()
    x_positions = np.arange(len(compare_columns))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6.8))
    palette = ['#C0392B', '#E69F00', '#2E8B57']

    for index, (_, row) in enumerate(top_df.iterrows()):
        ax.bar(
            x_positions + (index - 1) * width,
            row[compare_columns].to_numpy(dtype=float),
            width=width,
            color=palette[index % len(palette)],
            label=row['player_label'],
        )

    ax.plot(x_positions, average_values.values, color='black', marker='o', linewidth=2, label='Media do lote')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(compare_columns, rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Top Atletas vs Media do Lote', fontweight='bold')
    ax.set_ylabel('Sinal de risco normalizado')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def build_score_probability_scatter(results: pd.DataFrame):
    """Constroi um grafico de bolhas: score heuristico versus probabilidade de alto risco."""
    plot_df = _build_explanation_frame(results)

    if plot_df.empty or 'prob_high' not in plot_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Nenhum dado de score/probabilidade disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    risk_colors = {'Low': '#2E8B57', 'Medium': '#E69F00', 'High': '#C0392B'}
    colors = [risk_colors.get(label, '#95a5a6') for label in plot_df['risk_label']] if 'risk_label' in plot_df.columns else '#95a5a6'
    sizes = plot_df['confidence'].to_numpy(dtype=float) * 280 + 70 if 'confidence' in plot_df.columns else 120
    score_reference = 4.0
    probability_reference = 0.5

    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axvspan(score_reference, max(5, plot_df['risk_score'].max() + 1), color='#FDECEC', alpha=0.28)
    ax.axhspan(probability_reference, 1.0, color='#FFF3E0', alpha=0.22)
    ax.axvline(score_reference, color='#7F8C8D', linestyle='--', linewidth=1.4)
    ax.axhline(probability_reference, color='#7F8C8D', linestyle='--', linewidth=1.4)
    ax.scatter(plot_df['risk_score'], plot_df['prob_high'], c=colors, s=sizes, alpha=0.78, edgecolor='black', linewidth=1)

    for _, row in plot_df.head(min(5, len(plot_df))).iterrows():
        ax.annotate(row['player_label'], (row['risk_score'], row['prob_high']), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Score de risco heuristico')
    ax.set_ylabel('Probabilidade do modelo para Alto risco')
    ax.set_title('Mapa de Bolhas de Risco', fontweight='bold')
    ax.text(score_reference + 0.08, 0.97, 'Zonas de referencia: score >= 4 e prob_high >= 0.50', ha='left', va='top', fontsize=8, color='#566573')
    ax.set_xlim(0, max(5, plot_df['risk_score'].max() + 1))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Baixo', markerfacecolor=risk_colors['Low'], markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Medio', markerfacecolor=risk_colors['Medium'], markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Alto', markerfacecolor=risk_colors['High'], markeredgecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Tamanho da bolha = confianca', markerfacecolor='#D5D8DC', markeredgecolor='black', markersize=11),
    ]
    ax.legend(handles=legend_handles, title='Classe prevista', loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.tight_layout()
    return fig


def predict_injury_risk(
    general_health: float,
    energy_level: float,
    sleep_hours: float,
    sleep_quality: float,
    wakes_rested: float,
    stress_level: float,
    motivation: float,
    rpe: float,
    fatigue_level: float,
    post_discomfort: float,
    accurate_pass: float,
    missed_pass: float,
    interception: float,
    shot_on_goal: float,
    successful_dribble: float,
    tackle_won: float,
    foul_made: float,
    goal: float,
) -> tuple:
    """
    Prediz o risco de lesao a partir de dados de saude e desempenho.
    
    Args:
        Todos os indicadores de saude, fadiga e desempenho
        
    Returns:
        Tupla com (risk_label, confidence, probabilities_text)
    """
    if not model_loaded or predictor is None:
        return "Erro", 0.0, "Modelo nao carregado. Treine o modelo primeiro."
    
    try:
        # Monta dicionario de features
        features = {
            'general_health': general_health,
            'energy_level': energy_level,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'wakes_rested': wakes_rested,
            'stress_level': stress_level,
            'motivation': motivation,
            'rpe': rpe,
            'fatigue_level': fatigue_level,
            'post_discomfort': post_discomfort,
            'accurate_pass': accurate_pass,
            'missed_pass': missed_pass,
            'interception': interception,
            'shot_on_goal': shot_on_goal,
            'successful_dribble': successful_dribble,
            'tackle_won': tackle_won,
            'foul_made': foul_made,
            'goal': goal,
        }
        
        # Obtem predicao
        result = predictor.predict_risk(features)
        
        # Formata saida
        risk_label = result['risk_label']
        confidence = result['confidence']
        
        probs = result['probabilities']
        prob_text = (
            f"Baixo: {probs['Low']:.1%}\n"
            f"Medio: {probs['Medium']:.1%}\n"
            f"Alto: {probs['High']:.1%}"
        )
        
        return risk_label, confidence, prob_text
    
    except Exception as e:
        return "Erro", 0.0, f"Falha na predicao: {_sanitize_message(str(e))}"


def build_batch_results_plot(results: pd.DataFrame):
    """Constroi grafico de probabilidades empilhadas para resultados em lote."""
    if results.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Ainda nao ha resultados em lote.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    plot_columns = ['nickname', 'position', 'prob_low', 'prob_medium', 'prob_high']
    available_columns = [column for column in plot_columns if column in results.columns]
    plot_df = results.loc[:, available_columns].copy()

    if 'nickname' not in plot_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Grafico indisponivel para os resultados atuais.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    if 'position' in plot_df.columns:
        plot_df['player_label'] = plot_df['nickname'].astype(str) + ' (' + plot_df['position'].astype(str) + ')'
    else:
        plot_df['player_label'] = plot_df['nickname'].astype(str)

    plot_df = plot_df.sort_values('prob_high', ascending=False)

    fig_height = max(4, len(plot_df) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = {
        'prob_low': '#2E8B57',
        'prob_medium': '#E69F00',
        'prob_high': '#C0392B',
    }
    label_map = {
        'prob_low': 'Baixo',
        'prob_medium': 'Medio',
        'prob_high': 'Alto',
    }

    left = pd.Series(0.0, index=plot_df.index)
    for column_name in ['prob_low', 'prob_medium', 'prob_high']:
        ax.barh(
            plot_df['player_label'],
            plot_df[column_name],
            left=left,
            color=colors[column_name],
            label=label_map[column_name],
        )
        left = left + plot_df[column_name]

    ax.set_title('Probabilidades de Predicao em Lote por Atleta', fontweight='bold')
    ax.set_xlabel('Probabilidade prevista')
    ax.set_ylabel('Atleta')
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def build_batch_class_count_plot(results: pd.DataFrame):
    """Constroi grafico de contagem de classes para resultados em lote."""
    if results.empty or 'risk_label' not in results.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Nenhum resumo de classes disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    risk_order = ['Low', 'Medium', 'High']
    counts = results['risk_label'].value_counts().reindex(risk_order, fill_value=0)
    colors = ['#2E8B57', '#E69F00', '#C0392B']
    display_labels = ['Baixo', 'Medio', 'Alto']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(display_labels, counts.values, color=colors)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(int(value)),
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    ax.set_title('Contagem de Classes Previstas', fontweight='bold')
    ax.set_xlabel('Classe de risco')
    ax.set_ylabel('Numero de atletas')
    ax.grid(axis='y', alpha=0.3)

    upper_limit = max(1, counts.max())
    ax.set_ylim(0, upper_limit + max(1, upper_limit * 0.15))

    plt.tight_layout()
    return fig


def build_prob_high_ranking(results: pd.DataFrame):
    """Constroi ranking de prob_high por atleta."""
    if results.empty or 'prob_high' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de ranking disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    if 'nickname' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nomes de atletas ausentes.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    plot_df = results[['nickname', 'position', 'prob_high']].copy() if 'position' in results.columns else results[['nickname', 'prob_high']].copy()
    plot_df = plot_df.sort_values('prob_high', ascending=True)

    if 'position' in plot_df.columns:
        plot_df['label'] = plot_df['nickname'].astype(str) + ' (' + plot_df['position'].astype(str) + ')'
    else:
        plot_df['label'] = plot_df['nickname'].astype(str)

    fig_height = max(4, len(plot_df) * 0.4)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    colors = ['#2E8B57' if x < 0.33 else '#E69F00' if x < 0.66 else '#C0392B' for x in plot_df['prob_high']]
    ax.barh(plot_df['label'], plot_df['prob_high'], color=colors)
    ax.set_xlabel('Probabilidade de Alto risco')
    ax.set_ylabel('Atleta')
    ax.set_title('Ranking de Probabilidade de Alto Risco', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def build_probability_heatmap(results: pd.DataFrame):
    """Constroi mapa de calor das probabilidades Baixo/Medio/Alto por atleta."""
    if results.empty or 'prob_low' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de probabilidade disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    if 'nickname' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nomes de atletas ausentes.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    plot_df = results[['nickname', 'prob_low', 'prob_medium', 'prob_high']].copy()
    plot_df = plot_df.sort_values('prob_high', ascending=False)
    plot_df = plot_df.set_index('nickname')

    fig_height = max(4, len(plot_df) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    im = ax.imshow(plot_df.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(['Baixo', 'Medio', 'Alto'])
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title('Mapa de Calor de Probabilidades por Atleta', fontweight='bold')

    for i in range(len(plot_df)):
        for j in range(len(plot_df.columns)):
            value = plot_df.values[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=ax, label='Probabilidade')
    plt.tight_layout()
    return fig


def build_position_risk_count(results: pd.DataFrame):
    """Constroi barras empilhadas das classes de risco por posicao."""
    if results.empty or 'position' not in results.columns or 'risk_label' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de posicao disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    position_risk = pd.crosstab(results['position'], results['risk_label'], margins=False)
    risk_order = ['Low', 'Medium', 'High']
    position_risk = position_risk[[col for col in risk_order if col in position_risk.columns]]

    colors = {'Low': '#2E8B57', 'Medium': '#E69F00', 'High': '#C0392B'}
    color_list = [colors.get(col, '#95a5a6') for col in position_risk.columns]

    fig, ax = plt.subplots(figsize=(8, 4))
    position_risk.plot(kind='bar', stacked=True, ax=ax, color=color_list)
    ax.set_title('Distribuicao de Risco por Posicao', fontweight='bold')
    ax.set_xlabel('Posicao')
    ax.set_ylabel('Numero de atletas')
    ax.legend(title='Classe de risco', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def build_confidence_histogram(results: pd.DataFrame):
    """Constroi histograma da confianca das predicoes."""
    if results.empty or 'confidence' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de confianca disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results['confidence'], bins=10, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(results['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {results["confidence"].mean():.3f}')
    ax.axvline(results['confidence'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {results["confidence"].median():.3f}')
    ax.set_title('Distribuicao da Confianca das Predicoes', fontweight='bold')
    ax.set_xlabel('Confianca')
    ax.set_ylabel('Frequencia')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def build_risk_factors_plot(results: pd.DataFrame):
    """Constroi grafico dos fatores de risco observados no lote."""
    if results.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    factors = {}
    
    if 'rpe' in results.columns:
        factors['RPE >= 8'] = (results['rpe'] >= 8).sum()
    if 'fatigue_level' in results.columns:
        factors['Fadiga >= 7'] = (results['fatigue_level'] >= 7).sum()
    if 'sleep_hours' in results.columns:
        factors['Sono < 6h'] = (results['sleep_hours'] < 6).sum()
    if 'post_discomfort' in results.columns:
        factors['Desconforto pos'] = (results['post_discomfort'] == 1).sum()
    if 'stress_level' in results.columns:
        factors['Estresse alto'] = (results['stress_level'] == 3).sum()

    if not factors:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Nenhum dado de fator de risco disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = ['#C0392B' if v > 0 else '#95a5a6' for v in factors.values()]
    bars = ax.bar(factors.keys(), factors.values(), color=colors_bar, edgecolor='black')

    for bar, value in zip(bars, factors.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, str(int(value)), ha='center', va='bottom', fontweight='bold')

    ax.set_title('Fatores de Risco Observados no Lote', fontweight='bold')
    ax.set_ylabel('Numero de atletas')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def build_rpe_fatigue_scatter(results: pd.DataFrame):
    """Constroi dispersao de RPE vs Fadiga colorida por risco."""
    if results.empty or 'rpe' not in results.columns or 'fatigue_level' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Nenhum dado de RPE/Fadiga disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    if 'risk_label' not in results.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Nenhum dado de classe de risco disponivel.', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    risk_colors = {'Low': '#2E8B57', 'Medium': '#E69F00', 'High': '#C0392B'}
    colors = [risk_colors.get(label, '#95a5a6') for label in results['risk_label']]

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = (results['prob_high'] * 200 + 50) if 'prob_high' in results.columns else 100

    ax.scatter(results['rpe'], results['fatigue_level'], c=colors, s=sizes, alpha=0.6, edgecolor='black', linewidth=1)
    ax.set_xlabel('RPE (Esforco percebido)', fontsize=11)
    ax.set_ylabel('Nivel de fadiga', fontsize=11)
    ax.set_title('RPE vs Nivel de Fadiga (colorido por risco)', fontweight='bold')
    ax.grid(alpha=0.3)

    for label, color in risk_colors.items():
        ax.scatter([], [], c=color, s=100, label=label, edgecolor='black')
    ax.legend(title='Nivel de risco', loc='upper left')

    plt.tight_layout()
    return fig


def predict_batch_from_uploads(
    pre_file: str | None,
    post_file: str | None,
    match_file: str | None,
    match_id: float | None,
) -> tuple[pd.DataFrame, object, object, object, object, object, object, object, object, object, object, object, str]:
    """Prediz risco de lesao para varios atletas a partir de CSVs brutos enviados."""
    if not model_loaded or predictor is None:
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Modelo nao carregado. Treine o modelo primeiro.",
        )

    if not pre_file or not post_file or not match_file:
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Envie um arquivo CSV para pre, post e match.",
        )

    try:
        resolved_match_id = int(match_id) if match_id is not None else 1
        results = predictor.predict_batch_from_raw_csvs(
            pre_file=pre_file,
            post_file=post_file,
            match_file=match_file,
            match_id=resolved_match_id,
        )

        display_columns = [
            'nickname',
            'position',
            'match_id',
            'risk_label',
            'confidence',
            'prob_low',
            'prob_medium',
            'prob_high',
        ]
        available_columns = [column for column in display_columns if column in results.columns]
        display_results = results.loc[:, available_columns].copy()

        for column_name in ['confidence', 'prob_low', 'prob_medium', 'prob_high']:
            if column_name in display_results.columns:
                display_results[column_name] = display_results[column_name].round(4)

        plot_prob = build_batch_results_plot(display_results)
        plot_count = build_batch_class_count_plot(display_results)
        plot_ranking = build_prob_high_ranking(display_results)
        plot_heatmap = build_probability_heatmap(display_results)
        plot_position = build_position_risk_count(display_results)
        plot_confidence = build_confidence_histogram(display_results)
        plot_signal_heatmap = build_risk_signal_heatmap(results)
        plot_breakdown = build_heuristic_breakdown_plot(results)
        plot_radar = build_top_risk_radar(results)
        plot_compare = build_top_players_vs_average_plot(results)
        plot_scatter = build_score_probability_scatter(results)

        status = f"Predicoes geradas para {len(display_results)} atleta(s)."
        return (
            display_results,
            plot_prob,
            plot_count,
            plot_ranking,
            plot_heatmap,
            plot_position,
            plot_confidence,
            plot_signal_heatmap,
            plot_breakdown,
            plot_radar,
            plot_compare,
            plot_scatter,
            status,
        )
    except Exception as e:
        status = f"Falha na predicao em lote: {_sanitize_message(str(e))}"
        return (
            pd.DataFrame(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            status,
        )


# Cria a interface Gradio
with gr.Blocks(title="Classificador de Risco de Lesão no Futebol", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Classificador de Risco de Lesão no Futebol

        **Sistema para classificação de risco de lesão em atletas de futebol**

        Esta aplicação prediz o risco de lesão com base em avaliações de saúde pré-atividade,
        níveis de fadiga pós-atividade e estatísticas de desempenho em partida.

        **Categorias de Risco:**
        - **Baixo**: Risco mínimo de lesão
        - **Medio**: Risco moderado de lesão
        - **Alto**: Risco elevado de lesão
        """
    )
    
    with gr.Tabs():
        # Aba 1: Predicao
        with gr.Tab("Predicao"):
            with gr.Row():
                gr.Markdown("## Dados de Saude e Desempenho do Atleta")
            
            # Indicadores de saude pre-atividade
            with gr.Group():
                gr.Markdown("### Avaliacao de Saude Pre-Atividade")
                
                with gr.Row():
                    general_health = gr.Slider(
                        minimum=1, maximum=4, step=1, value=2,
                        label="Saude geral (1=Ruim, 4=Excelente)"
                    )
                    energy_level = gr.Slider(
                        minimum=1, maximum=3, step=1, value=2,
                        label="Nivel de energia (1=Baixo, 3=Alto)"
                    )
                
                with gr.Row():
                    sleep_hours = gr.Slider(
                        minimum=0, maximum=12, step=0.5, value=7,
                        label="Horas de sono (media dos ultimos 7 dias)"
                    )
                    sleep_quality = gr.Slider(
                        minimum=1, maximum=3, step=1, value=2,
                        label="Qualidade do sono (1=Ruim, 3=Boa)"
                    )
                
                with gr.Row():
                    wakes_rested = gr.Slider(
                        minimum=0, maximum=3, step=1, value=2,
                        label="Acorda descansado (0=Nunca, 3=Sempre)"
                    )
                    stress_level = gr.Slider(
                        minimum=1, maximum=3, step=1, value=2,
                        label="Nivel de estresse (1=Baixo, 3=Alto)"
                    )
                
                motivation = gr.Slider(
                    minimum=1, maximum=3, step=1, value=2,
                    label="Motivacao (1=Baixa, 3=Alta)"
                )
            
            # Indicadores pos-atividade
            with gr.Group():
                gr.Markdown("### Avaliacao Pos-Atividade")
                
                with gr.Row():
                    rpe = gr.Slider(
                        minimum=1, maximum=10, step=1, value=6,
                        label="RPE - Esforco percebido (1-10)"
                    )
                    fatigue_level = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5,
                        label="Nivel de fadiga (1-10)"
                    )
                
                post_discomfort = gr.Slider(
                    minimum=0, maximum=1, step=1, value=0,
                    label="Desconforto pos-atividade (0=Nenhum, 1=Sim)"
                )
            
            # Metricas de desempenho da partida
            with gr.Group():
                gr.Markdown("### Metricas de Desempenho na Partida")
                
                with gr.Row():
                    accurate_pass = gr.Number(value=50, label="Passes certos")
                    missed_pass = gr.Number(value=10, label="Passes errados")
                
                with gr.Row():
                    interception = gr.Number(value=3, label="Interceptacoes")
                    shot_on_goal = gr.Number(value=2, label="Finalizacoes no gol")
                
                with gr.Row():
                    successful_dribble = gr.Number(value=3, label="Dribles bem-sucedidos")
                    tackle_won = gr.Number(value=2, label="Desarmes ganhos")
                
                with gr.Row():
                    foul_made = gr.Number(value=1, label="Faltas cometidas")
                    goal = gr.Number(value=0, label="Gols marcados")
            
            # Botao de predicao
            predict_button = gr.Button("Predizer risco de lesao", size="lg", scale=2)
            
            # Resultados
            with gr.Group():
                gr.Markdown("### Resultado da Predicao")
                
                with gr.Row():
                    risk_output = gr.Label(label="Classificacao de risco", scale=1)
                    confidence_output = gr.Number(label="Confianca", scale=1)
                
                prob_output = gr.Textbox(
                    label="Probabilidades de risco",
                    interactive=False,
                    lines=3
                )
            
            # Conecta botao a funcao de predicao
            predict_button.click(
                fn=predict_injury_risk,
                inputs=[
                    general_health, energy_level, sleep_hours, sleep_quality,
                    wakes_rested, stress_level, motivation, rpe, fatigue_level,
                    post_discomfort, accurate_pass, missed_pass, interception,
                    shot_on_goal, successful_dribble, tackle_won, foul_made, goal
                ],
                outputs=[risk_output, confidence_output, prob_output]
            )

        with gr.Tab("Envio em Lote"):
            gr.Markdown(
                """
                ## Predição em Lote com Upload de CSV Bruto

                Envie um CSV bruto de cada fonte:
                - `pre`: questionário coletado antes da partida
                - `post`: questionário coletado depois da partida
                - `match`: exportação de estatísticas da partida

                Se os arquivos enviados não tiverem `match_id`, o mesmo valor abaixo
                será aplicado aos três uploads para que sejam mesclados como um lote único.
                """
            )

            with gr.Row():
                pre_batch_file = gr.File(
                    label="CSV Pre",
                    file_types=[".csv"],
                    type="filepath",
                )
                post_batch_file = gr.File(
                    label="CSV Post",
                    file_types=[".csv"],
                    type="filepath",
                )
                match_batch_file = gr.File(
                    label="CSV Match",
                    file_types=[".csv"],
                    type="filepath",
                )

            batch_match_id = gr.Number(
                value=1,
                precision=0,
                label="ID da partida (opcional)",
            )

            batch_predict_button = gr.Button("Executar predição em lote", size="lg")

            batch_results_output = gr.Dataframe(
                label='Resultados da predição em lote',
                interactive=False,
            )

            with gr.Tabs():
                with gr.Tab("Resumo e Probabilidades"):
                    with gr.Row():
                        batch_class_count_plot = gr.Plot(label='Contagem de classes')
                        batch_prob_ranking = gr.Plot(label='Ranking de alto risco')
                    with gr.Row():
                        batch_results_plot = gr.Plot(label='Probabilidades empilhadas')

                with gr.Tab("Perfil"):
                    batch_position_plot = gr.Plot(label='Risco por posicao')
                    batch_confidence_plot = gr.Plot(label='Distribuicao de confianca')

                with gr.Tab("Analises"):
                    with gr.Row():
                        batch_signal_heatmap_plot = gr.Plot(label='Mapa de calor de sinais de risco')
                        batch_heatmap_plot = gr.Plot(label='Mapa de calor de probabilidades')
                    with gr.Row():
                        batch_breakdown_plot = gr.Plot(label='Detalhamento do score heuristico')
                        batch_radar_plot = gr.Plot(label='Radar de maior risco')
                    with gr.Row():
                        batch_compare_plot = gr.Plot(label='Top atletas vs media do lote')
                        batch_score_scatter_plot = gr.Plot(label='Mapa de bolhas de risco')

            batch_status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )

            batch_predict_button.click(
                fn=predict_batch_from_uploads,
                inputs=[pre_batch_file, post_batch_file, match_batch_file, batch_match_id],
                outputs=[
                    batch_results_output,
                    batch_results_plot,
                    batch_class_count_plot,
                    batch_prob_ranking,
                    batch_heatmap_plot,
                    batch_position_plot,
                    batch_confidence_plot,
                    batch_signal_heatmap_plot,
                    batch_breakdown_plot,
                    batch_radar_plot,
                    batch_compare_plot,
                    batch_score_scatter_plot,
                    batch_status_output,
                ],
            )
        
        # Aba 2: Informacoes
        with gr.Tab("Sobre"):
            gr.Markdown(
                """
                ## Sobre este sistema
                
                ### Objetivo
                Classificar o risco de lesão de atletas de futebol usando aprendizado de máquina
                com base em indicadores de saúde, fadiga e desempenho.
                
                ### Fontes de dados
                - **Pré-atividade**: Avaliações de saúde, qualidade do sono, níveis de estresse
                - **Pós-atividade**: Esforço percebido (RPE), fadiga, desconforto
                - **Desempenho na partida**: Passes, finalizações, desarmes, gols
                
                ### Modelo
                - **Algoritmo**: Multíplos modelos treinados (Regressão Logística, Random Forest, SVM)
                - **Validação**: Validação cruzada estratificada com 3 folds
                - **Classes alvo**: Risco baixo, médio e alto
                
                ### Limiares de risco
                O score de risco de lesão e computado de forma aditiva:
                - Desconforto pós-atividade: +2 pontos
                - RPE >= 8: +1 ponto
                - Fadiga >= 7: +1 ponto
                - Desconforto pré-atividade: +1 ponto
                - Sono < 6 horas: +1 ponto
                - Estresse alto: +1 ponto
                - Qualidade do sono ruim: +1 ponto
                
                **Classificação:**
                - 0-1 pontos -> risco baixo
                - 2-3 pontos -> risco medio
                - 4+ pontos -> risco alto
                
                ### Aviso
                Este sistema é apenas para **fins de pesquisa e educacionais**.
                Ele não substitui avaliação médica profissional nem
                avaliações de medicina esportiva.
                """
            )
        
        # Aba 3: Instrucoes
        with gr.Tab("Instrucoes"):
            gr.Markdown("## Como usar")

            with gr.Row():
                gr.Markdown(
                    """
                    ### Aba Predicao

                    #### Passo 1: Insira dados de saúde pré-atividade
                    - Avalie saúde geral, energia e qualidade do sono da última semana
                    - Informe nível de estresse e motivação

                    #### Passo 2: Insira avaliação pós-atividade
                    - Registre o esforco percebido (RPE) em escala de 1 a 10
                    - Informe o nível atual de fadiga
                    - Indique se houve desconforto durante a atividade

                    #### Passo 3: Insira métricas de desempenho
                    - Informe estatísticas da partida (passes, finalizações, desarmes etc.)
                    - Use contagens reais ou estimativas consistentes

                    #### Passo 4: Execute a predição
                    - Clique no botao "Predizer risco de lesão"
                    - Revise classificação de risco, confiança e probabilidades

                    #### Saída
                    - **Rótulo de risco**: Baixo, Medio ou Alto
                    - **Confiança**: Probabilidade da classe prevista
                    - **Probabilidades**: Distribuição em todas as categorias de risco
                    """
                )

                gr.Markdown(
                    """
                    ### Aba Envio em Lote

                    #### Passo 1: Envie os arquivos CSV brutos
                    - Envie um arquivo em **CSV Pre**
                    - Envie um arquivo em **CSV Post**
                    - Envie um arquivo em **CSV Match**

                    #### Passo 2: Defina o Match ID
                    - Use o campo opcional **Match ID** quando os CSVs não tiverem essa coluna
                    - O mesmo valor será aplicado para mesclar os três arquivos em um único lote

                    #### Passo 3: Execute o processamento em lote
                    - Clique no botão "Executar predição em lote"
                    - Aguarde o carregamento da tabela, dos gráficos e da mensagem de status

                    #### Saída
                    - Tabela **Resultados da predição em lote** para todos os atletas enviados
                    - Gráficos de **Resumo e Probabilidades** para todo o lote
                    - Gráficos de **Perfil** com visão de classe e confiança
                    - Gráficos de **Análises** para explicar por que atletas aparecem com risco maior ou menor
                    """
                )

            gr.Markdown(
                """
                ### Recomendações
                - **Baixo risco**: Continue treino e atividades normalmente
                - **Médio risco**: Monitore de perto e considere reduzir a carga
                - **Alto risco**: Consulte especialista em medicina esportiva e considere repouso
                """
            )


# Inicia a interface
if __name__ == "__main__":
    print("Classificador de Risco de Lesão no futebol - Interface Gradio")
    print("\nIniciando servidor Gradio...")
    print("Abra o navegador na URL exibida abaixo\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
