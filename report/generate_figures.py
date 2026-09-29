"""Generate report figures through the user's existing runs/plot.py.

Only experiment paths, English labels, output paths, and figure size are adapted.
The plotting and data-reading functions are the original script's functions.
"""
from pathlib import Path
import importlib.util
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'report' / 'figures'
FIG.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('original_run_plot', ROOT/'runs'/'plot.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
original_subplots = plot.plt.subplots

def publication_size(*args, **kwargs):
    # Same original plot, scaled to keep labels legible in a full-width figure.
    kwargs['figsize'] = (10, 3.9)
    return original_subplots(*args, **kwargs)

plot.plt.subplots = publication_size
plot.X_AXIS = 'step'
groups = [
    ('earlier_recipe.png', [('s_direct', 'Direct CE'), ('s_kd', 'Distillation')]),
    ('revised_recipe.png', [('s_dir48', 'Direct CE'), ('s_both48', 'Distillation')]),
    ('architecture_curves.png', [('ab_main4800', 'Reference'),
                                 ('ab_gelu4800', 'GELU'),
                                 ('ab_drop4800', 'No dropout'),
                                 ('ab_untied4800', 'Untied')]),
]
for filename, experiments in groups:
    plot.OUTPUT_PLOT_PATH = str(FIG/filename)
    plot.plot_separate_metrics([
        {'path': str(ROOT/'runs'/name/'metrics.json'), 'label': label}
        for name, label in experiments
    ])
    plot.plt.close('all')
print('Generated three figures using runs/plot.py.')
