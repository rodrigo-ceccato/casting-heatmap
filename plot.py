#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from difflib import get_close_matches          #  ### NEW

# ——— CONFIG ———
CSV_PATH   = 'Casting.csv'
NAME_COL   = 'Seu nome'
OUTPUT_PDF = 'casting_heatmap_with_table.pdf'

# — auto-detect preference columns —
cols       = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
FIRST_COL  = next(c for c in cols if c.startswith('Primeira opção'))
SECOND_COL = next(c for c in cols if c.startswith('Segunda opção'))
THIRD_COL  = next(c for c in cols if c.startswith('Terceira opção'))
REFUSE_COL = next(c for c in cols if c.startswith('Não quero'))

# — load data & identify roles / actors —
df           = pd.read_csv(CSV_PATH)
casting_cols = [c for c in df.columns if c.startswith('Casting [')]
roles        = [c[len('Casting ['):-1] for c in casting_cols]
actors       = sorted(set(df[NAME_COL].str.strip().str.split().str[0])  # first names
                      | set(df_long := df[casting_cols].melt().value))   # any extras

# --- helper: fuzzy map a free-typed role to the canonical spelling ---
def match_role(txt):
    """Return canonical role name or None."""
    if pd.isna(txt):
        return None
    txt = str(txt).strip()
    if txt in roles:
        return txt
    hit = get_close_matches(txt, roles, n=1, cutoff=0.8)
    return hit[0] if hit else None

# — build pivot for heat-map (same orientation as before) —
df_long = df[[NAME_COL] + casting_cols].melt(id_vars=NAME_COL,
                                             var_name='role_col',
                                             value_name='actor')
df_long['role'] = df_long['role_col'].str.extract(r'Casting \[(.*)\]')
pivot = (df_long.groupby(['role', 'actor'])
                 .size()
                 .unstack(fill_value=0)
                 .reindex(index=roles)
                 .sort_index(axis=1))

def plot_and_save(df, pivot, roles, actors,
                  NAME_COL,
                  FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
                  output_pdf):

    symbols = {
        FIRST_COL : ('★', 'gold',   16),
        SECOND_COL: ('☆', 'orange', 14),
        THIRD_COL : ('✩', 'gray',   12),
        REFUSE_COL: ('✘', 'red',    18),
    }

    # 1) Build preference grid *transposed*  ### CHANGED
    pref_grid = pd.DataFrame('', index=roles, columns=actors)
    for _, row in df.iterrows():
        actor_col = row[NAME_COL].strip().split()[0]      # actor column key
        if actor_col not in actors:
            continue
        # ★, ☆, ✩
        for col in (FIRST_COL, SECOND_COL, THIRD_COL):
            role = match_role(row[col])
            if role:
                pref_grid.at[role, actor_col] += symbols[col][0]
        # ✘
        nr = match_role(row[REFUSE_COL])
        if nr:
            pref_grid.at[nr, actor_col] += symbols[REFUSE_COL][0]

    # 2) Figure layout
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 3], hspace=0.35)

    # ─── TOP: table ───
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    tbl = ax0.table(cellText=pref_grid.values,
                    rowLabels=pref_grid.index,
                    colLabels=pref_grid.columns,
                    cellLoc='center',
                    loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)                              #  ### CHANGED (smaller)
    tbl.scale(1, 1.8)

    # keep column widths reasonable
    try:
        tbl.auto_set_column_width(col=list(range(len(actors)+1)))
    except Exception:
        pass

    # clip long names instead of spilling              ### NEW
    for cell in tbl.get_celld().values():
        cell.get_text().set_clip_on(True)

    # colour the texts
    for r, role in enumerate(pref_grid.index):
        for c, actor in enumerate(pref_grid.columns):
            txt = pref_grid.iat[r, c]
            for (_sym, color, _sz) in symbols.values():
                if _sym in txt:
                    tbl[r+1, c].get_text().set_color(color)
                    break

    ax0.set_title('Preferências (Transposta) – Linhas: Personagens, Colunas: Atores',
                  pad=10)

    # ─── BOTTOM: heat-map ───
    ax1 = fig.add_subplot(gs[1])
    max_val = int(pivot.values.max())
    im = ax1.imshow(pivot.values,
                    cmap=plt.get_cmap('viridis', max_val + 1),
                    vmin=0, vmax=max_val,
                    aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax1,
                        ticks=range(max_val + 1),
                        label='Nº de escolhas')
    cbar.ax.set_yticklabels(map(str, range(max_val + 1)))

    # grid
    ax1.set_xticks(np.arange(-.5, len(pivot.columns), 1), minor=True)
    ax1.set_yticks(np.arange(-.5, len(roles), 1),        minor=True)
    ax1.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax1.tick_params(which="minor", bottom=False, left=False)

    # axis labels – still no rotation
    ax1.set_xticks(np.arange(len(pivot.columns)))
    ax1.set_xticklabels(pivot.columns, rotation=0, ha='center', fontsize=8)
    ax1.set_yticks(np.arange(len(roles)))
    ax1.set_yticklabels(roles, fontsize=9)
    ax1.set_title('Mapa de Calor (discreto) com Sobreposição de Preferências')

    # ─── Overlay ★/☆/✩/✘ – fuzzy-match fixed  ### CHANGED
    actor_to_x = {a: i for i, a in enumerate(pivot.columns)}
    role_to_y  = {r: i for i, r in enumerate(roles)}

    rng = np.random.default_rng()  # better random
    for _, row in df.iterrows():
        actor_col = row[NAME_COL].strip().split()[0]
        if actor_col not in actor_to_x:
            continue
        x = actor_to_x[actor_col]

        # stars
        for pref_col in (FIRST_COL, SECOND_COL, THIRD_COL):
            role = match_role(row[pref_col])
            if role:
                y = role_to_y[role]
                sym, color, sz = symbols[pref_col]
                dx, dy = (rng.random(2) - 0.5) * 0.3
                ax1.text(x + dx, y + dy, sym,
                         ha='center', va='center',
                         color=color, fontsize=sz, weight='bold')

        # X
        nr_role = match_role(row[REFUSE_COL])
        if nr_role:
            y = role_to_y[nr_role]
            sym, color, sz = symbols[REFUSE_COL]
            dx, dy = (rng.random(2) - 0.5) * 0.3
            ax1.text(x + dx, y + dy, sym,
                     ha='center', va='center',
                     color=color, fontsize=sz, weight='bold')

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

# — run —
plot_and_save(df, pivot, roles, actors,
              NAME_COL,
              FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
              OUTPUT_PDF)
