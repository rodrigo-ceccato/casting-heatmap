#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ——— CONFIG ———
CSV_PATH = 'Casting.csv'   # ← your real CSV filename
NAME_COL = 'Seu nome'

# auto-detect the exact column names to avoid KeyErrors
cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
FIRST_COL   = next(c for c in cols if c.startswith('Primeira opção'))
SECOND_COL  = next(c for c in cols if c.startswith('Segunda opção'))
THIRD_COL   = next(c for c in cols if c.startswith('Terceira opção'))
REFUSE_COL  = next(c for c in cols if c.startswith('Não quero'))

# load full data
df = pd.read_csv(CSV_PATH)

# find all “Casting [Role]” columns
casting_cols = [c for c in df.columns if c.startswith('Casting [')]
roles = [c[len('Casting ['):-1] for c in casting_cols]

# ——— 1. LONG-FORM FOR HEATMAP ———
df_long = (
    df[[NAME_COL] + casting_cols]
    .melt(id_vars=NAME_COL, var_name='role_col', value_name='actor')
)
df_long['role'] = df_long['role_col'].str.extract(r'Casting \[(.*)\]')

# pivot to count picks
pivot = (
    df_long
    .groupby(['role','actor'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=roles)           # preserve original role order
    .sort_index(axis=1)             # sort actors alphabetically
)

# ——— 2. BUILD PREFERENCE GRID (PERSON×ROLE) ———
symbols = {
    FIRST_COL : ('★', 'gold'),
    SECOND_COL: ('☆', 'orange'),
    THIRD_COL : ('✩', 'gray'),
    REFUSE_COL: ('✘', 'red'),
}

# empty grid of strings
pref_grid = pd.DataFrame('', index=df[NAME_COL], columns=roles)

for _, row in df.iterrows():
    person = row[NAME_COL]
    # stars
    for col in (FIRST_COL, SECOND_COL, THIRD_COL):
        role = row[col]
        if pd.notna(role) and role in roles:
            symbol, _ = symbols[col]
            pref_grid.at[person, role] += symbol
    # refusal
    role_no = row[REFUSE_COL]
    if pd.notna(role_no) and role_no in roles:
        sym, _ = symbols[REFUSE_COL]
        pref_grid.at[person, role_no] += sym

# ——— 3. PLOT EVERYTHING ———
fig = plt.figure(figsize=(14,12))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.3)

# — preference grid on top
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
tbl.set_fontsize(11)
tbl.scale(1, 1.4)

# color each cell’s text by the **last** symbol it contains
for i, person in enumerate(pref_grid.index):
    for j, role in enumerate(pref_grid.columns):
        txt = pref_grid.at[person, role]
        for colname, (sym, color) in symbols.items():
            if sym in txt:
                tbl[i+1, j].get_text().set_color(color)
                break

ax0.set_title('Preferências por Respondente (★=1ª ☆=2ª ✩=3ª   ✘=Não quero)', pad=12)

# — heatmap below
ax1 = fig.add_subplot(gs[1])
im = ax1.imshow(pivot.values, aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Nº de escolhas')

ax1.set_xticks(np.arange(len(pivot.columns)))
ax1.set_xticklabels(pivot.columns, rotation=45, ha='right')
ax1.set_yticks(np.arange(len(pivot.index)))
ax1.set_yticklabels(pivot.index)
ax1.set_title('Mapa de calor: frequência de escolhas por ator/role')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap_with_overlay(df, pivot, roles,
                              FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
                              casting_cols):
    """
    1) Draws the actor×role heatmap (pivot)
    2) Overlays each respondent's ★/☆/✩ at (actor,role) where they cast that actor
       and ✘ where they refused the role.
    """
    # symbol → (marker text, color, size)
    symbols = {
        FIRST_COL : ('★', 'gold',   16),
        SECOND_COL: ('☆', 'orange', 14),
        THIRD_COL : ('✩', 'gray',   12),
        REFUSE_COL: ('✘', 'red',    16),
    }

    # build a lookup for actor→x-position
    actors = pivot.columns.tolist()
    actor_to_x = {actor: xi for xi, actor in enumerate(actors)}
    # roles → y-position
    role_to_y  = {role: yi for yi, role in enumerate(roles)}

    # 1. plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, label='Nº de escolhas')

    ax.set_xticks(np.arange(len(actors)))
    ax.set_xticklabels(actors, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(roles)))
    ax.set_yticklabels(roles)
    ax.set_title('Heatmap de Atribuições + Sobreposição de Preferências')

    # 2. overlay every respondent’s symbol
    for _, row in df.iterrows():
        person = row['Seu nome']
        # for each role, find which actor they assigned
        for col in casting_cols:
            role = col[len('Casting ['):-1]
            actor = row[col]
            if pd.isna(actor) or role not in role_to_y:
                continue

            # decide which symbol (if any) this respondent has on that role
            if row[FIRST_COL] == role:
                sym, color, sz = symbols[FIRST_COL]
            elif row[SECOND_COL] == role:
                sym, color, sz = symbols[SECOND_COL]
            elif row[THIRD_COL] == role:
                sym, color, sz = symbols[THIRD_COL]
            elif row[REFUSE_COL] == role:
                sym, color, sz = symbols[REFUSE_COL]
            else:
                continue

            x = actor_to_x[actor]
            y = role_to_y[role]
            # slight jitter so overlapping symbols spread out
            jitter = (np.random.rand(2) - 0.5) * 0.3
            ax.text(x + jitter[0],
                    y + jitter[1],
                    sym,
                    ha='center',
                    va='center',
                    color=color,
                    fontsize=sz,
                    weight='bold')

    plt.tight_layout()
    plt.show()

plot_heatmap_with_overlay(
    df, pivot, roles,
    FIRST_COL, SECOND_COL, THIRD_COL, REFUSE_COL,
    casting_cols
)
