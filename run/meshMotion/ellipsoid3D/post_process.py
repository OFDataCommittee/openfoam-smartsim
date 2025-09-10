import os
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1. Paths & read ─────────────────────────────────────────────────────────
base_dir   = 'postProcessing'
sub_dir    = 'fieldMinMax'
case_dir   = '0'
filename   = 'fieldMinMax.dat'
data_dir   = os.path.join(base_dir, sub_dir, case_dir)
dat_path   = os.path.join(data_dir, filename)
csv_path   = os.path.join(data_dir, 'fieldMinMax.csv')
pdf_path   = os.path.join(data_dir, 'fieldMinMax_plots.pdf')

df = pd.read_table(
    dat_path,
    sep='\t',
    header=None,
    skiprows=[0,1],
    names=[
        'time',
        'field',
        'min',
        'location_min',
        'processor_min',
        'max',
        'location_max',
        'processor_max'
    ]
)

# ─── 2. Clean & compute mean ─────────────────────────────────────────────────
df['field'] = df['field'].str.strip()
df['mean']  = (df['min'] + df['max']) / 2.0

# ─── 3. Export cleaned CSV (no index) ────────────────────────────────────────
df.to_csv(csv_path, index=False)

# ─── 4. Select the two fields ───────────────────────────────────────────────
df_skew   = df[df['field'] == 'skewness']
df_nonortho = df[df['field'] == 'nonOrthoAngle']

# ─── 5. Plot setup ─────────────────────────────────────────────────────────
plt.rc('font', size=11)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=False)

# common plotting function
def plot_field(ax, data, name):
    ax.plot(data['time'], data['min'],  marker='o', linestyle='-', label=f'{name} min')
    ax.plot(data['time'], data['mean'],marker='o', linestyle='--',label=f'{name} mean')
    ax.plot(data['time'], data['max'],  marker='o', linestyle=':', label=f'{name} max')
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel(name)
    ax.grid(True)
    ax.legend(fontsize=9)

# ─── 6. Draw the two panels ─────────────────────────────────────────────────
plot_field(axes[0], df_skew,       'skewness')
plot_field(axes[1], df_nonortho, 'nonOrthoAngle')

fig.tight_layout()

# ─── 7. Save to PDF ─────────────────────────────────────────────────────────
fig.savefig(pdf_path, format='pdf')
print(f"Saved plots to {pdf_path}")