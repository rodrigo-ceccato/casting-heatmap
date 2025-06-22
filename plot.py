#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ——— CONFIG ———
CSV_PATH = 'Casting.csv'    # ← your CSV file
NAME_COL = 'Seu nome'

# — auto-detect the preference columns to avoid typos/KeyErrors —
cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
FIRST_COL  = next(c for c in cols if c.startswith('Primeira opção'))
SECOND_COL = next(c for c in cols if c.startswith('Segunda opção'))
THIRD_COL  = next(c for c in cols if c.startswith('Terceira opção'))
REFUSE_COL = next(c for c in cols if c.startswith('Não quero'))

# — load full data —
df = pd.read_csv(CSV_PATH)

# — identify all “Casting [Role]” columns —
casting_cols = [c for c in df.columns if c.startswith('Casting [')]
roles = [c[len('Casting ['):-1] for c in casting_cols]

# — build the pivot (heatmap) —
df_long = df[[NAME_COL] + casting_cols].melt(
    id_vars=NAME_COL, var_name='role_col', value_name='actor'
)
df_long['role'] = df_long['role_col'].str.extract(r'Casting \[(.*)\]')
pivot = (
    df_long
    .groupby(['role','actor'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=roles)      # keep original role order
    .sort_index(axis=1)        # actors alpha-sorted
)

def plot_heatmap_with_table_and_overlay(
    df, pivot, roles,
    NAME_COL,
    FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
    casting_cols
):
    # symbol → (marker, color, font size)
    symbols = {
        FIRST_COL : ('★', 'gold',   16),
        SECOND_COL: ('☆', 'orange', 14),
        THIRD_COL : ('✩', 'gray',   12),
        REFUSE_COL: ('✘', 'red',    18),
    }

    # actor→x-index, role→y-index
    actors    = pivot.columns.tolist()
    actor_to_x = {a:i for i,a in enumerate(actors)}
    role_to_y  = {r:i for i,r in enumerate(roles)}

    # — build preference-table strings —
    pref_grid = pd.DataFrame('', index=df[NAME_COL], columns=roles)
    for _, row in df.iterrows():
        person = row[NAME_COL]
        # ★/☆/✩
        for col in (FIRST_COL, SECOND_COL, THIRD_COL):
            role = row[col]
            if pd.notna(role) and role in roles:
                pref_grid.at[person, role] += symbols[col][0]
        # ✘ in the table too
        no_role = row[REFUSE_COL]
        if pd.notna(no_role) and no_role in roles:
            pref_grid.at[person, no_role] += symbols[REFUSE_COL][0]

    # — set up the figure with a table on top, heatmap below —
    fig = plt.figure(figsize=(14,12))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.3)

    # ─── top: preference table ───
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    tbl = ax0.table(
        cellText=pref_grid.values,
        rowLabels=pref_grid.index,
        colLabels=pref_grid.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)
    # color each cell by its last symbol
    for i, person in enumerate(pref_grid.index):
        for j, role in enumerate(pref_grid.columns):
            txt = pref_grid.at[person, role]
            for colname, (_sym, color, _sz) in symbols.items():
                if _sym in txt:
                    tbl[i+1, j].get_text().set_color(color)
                    break
    ax0.set_title(
        'Preferências por Respondente (★=1ª  ☆=2ª  ✩=3ª   ✘=Não quero)', 
        pad=12
    )

    # ─── bottom: heatmap + overlay ───
    ax1 = fig.add_subplot(gs[1])
    im = ax1.imshow(pivot.values, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Nº de escolhas')

    ax1.set_xticks(np.arange(len(actors)))
    ax1.set_xticklabels(actors, rotation=45, ha='right')
    ax1.set_yticks(np.arange(len(roles)))
    ax1.set_yticklabels(roles)
    ax1.set_title('Mapa de Calor com Sobreposição de Preferências')

    # → overlay ★/☆/✩ at the actor they actually cast
    for _, row in df.iterrows():
        for pref_col in (FIRST_COL, SECOND_COL, THIRD_COL):
            role = row[pref_col]
            if pd.isna(role) or role not in roles:
                continue
            col_name = f'Casting [{role}]'
            if col_name not in df.columns:
                continue
            actor = row[col_name]
            if pd.isna(actor):
                continue

            sym, color, sz = symbols[pref_col]
            x = actor_to_x.get(actor)
            y = role_to_y[role]
            if x is None:
                continue
            dx, dy = (np.random.rand(2) - 0.5) * 0.3
            ax1.text(x+dx, y+dy, sym,
                     ha='center', va='center',
                     color=color, fontsize=sz, weight='bold')

    # → overlay ✘ at **their own actor-column** for the role they refused
    for _, row in df.iterrows():
        no_role = row[REFUSE_COL]
        if pd.isna(no_role) or no_role not in roles:
            continue
        # assume respondent’s first name matches column
        resp = row[NAME_COL].strip().split()[0]
        x = actor_to_x.get(resp)
        y = role_to_y[no_role]
        if x is None:
            continue
        sym, color, sz = symbols[REFUSE_COL]
        dx, dy = (np.random.rand(2) - 0.5) * 0.3
        ax1.text(x+dx, y+dy, sym,
                 ha='center', va='center',
                 color=color, fontsize=sz, weight='bold')

    plt.tight_layout()
    plt.show()

# — invoke the plot —
plot_heatmap_with_table_and_overlay(
    df, pivot, roles,
    NAME_COL,
    FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
    casting_cols
)
