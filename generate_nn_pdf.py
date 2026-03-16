import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────────────
C_INPUT   = '#3498DB'   # blue
C_HIDDEN  = '#E67E22'   # orange
C_OUTPUT  = '#2ECC71'   # green
C_WEIGHT  = '#9B59B6'   # purple
C_DARK    = '#2C3E50'   # dark bg
C_RED     = '#E74C3C'   # accent red
C_WHITE   = '#FFFFFF'
C_LGRAY   = '#ECF0F1'
C_DGRAY   = '#7F8C8D'
C_YELLOW  = '#F1C40F'
C_TEAL    = '#1ABC9C'

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def hex_rgba(h, a=1.0):
    r, g, b = hex_to_rgb(h)
    return (r, g, b, a)

OUTPUT_PATH = r'C:\Users\admin\VSCode\NeuralNets\NeuralNetwork_Documentation.pdf'

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title Page
# ══════════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C_DARK)

    # Gradient-like bands
    for i, (y0, y1, alpha) in enumerate([(0, 0.12, 0.6), (0.88, 1.0, 0.6)]):
        band = plt.Rectangle((0, y0), 1, y1-y0,
                              color=hex_rgba(C_INPUT, alpha), zorder=0)
        ax.add_patch(band)

    # Decorative circles (neuron-like)
    rng = np.random.default_rng(42)
    for _ in range(30):
        x, y = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)
        r = rng.uniform(0.005, 0.018)
        c = rng.choice([C_INPUT, C_HIDDEN, C_OUTPUT, C_WEIGHT])
        circle = plt.Circle((x, y), r, color=hex_rgba(c, 0.15), zorder=1)
        ax.add_patch(circle)

    # Title
    ax.text(0.5, 0.76, 'Neural Network for',
            ha='center', va='center', fontsize=28, fontweight='bold',
            color=C_LGRAY, zorder=5)
    ax.text(0.5, 0.65, 'MNIST Digit Classification',
            ha='center', va='center', fontsize=36, fontweight='bold',
            color=C_WHITE, zorder=5)

    # Subtitle bar
    bar = FancyBboxPatch((0.1, 0.56), 0.8, 0.045,
                         boxstyle='round,pad=0.01',
                         facecolor=hex_rgba(C_INPUT, 0.4),
                         edgecolor=hex_rgba(C_INPUT, 0.9), linewidth=2, zorder=4)
    ax.add_patch(bar)
    ax.text(0.5, 0.582, 'A from-scratch NumPy implementation of a 2-layer feedforward neural network',
            ha='center', va='center', fontsize=11, color=C_WHITE, zorder=5)

    # Architecture summary boxes
    arch_items = [
        ('Input Layer', 'A[0] = X\n784 units\n28×28 pixels', C_INPUT),
        ('Hidden Layer', 'A[1]\n10 units\nReLU activation', C_HIDDEN),
        ('Output Layer', 'A[2]\n10 units\nSoftmax (digits 0–9)', C_OUTPUT),
    ]
    for i, (title, body, color) in enumerate(arch_items):
        x0 = 0.06 + i * 0.315
        box = FancyBboxPatch((x0, 0.32), 0.27, 0.19,
                             boxstyle='round,pad=0.015',
                             facecolor=hex_rgba(color, 0.25),
                             edgecolor=hex_rgba(color, 0.9), linewidth=2, zorder=4)
        ax.add_patch(box)
        ax.text(x0 + 0.135, 0.49, title,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=color, zorder=5)
        ax.text(x0 + 0.135, 0.40, body,
                ha='center', va='center', fontsize=9.5, color=C_LGRAY,
                linespacing=1.5, zorder=5)
        if i < 2:
            ax.annotate('', xy=(x0 + 0.315, 0.415), xytext=(x0 + 0.27, 0.415),
                        arrowprops=dict(arrowstyle='->', color=C_WEIGHT, lw=2), zorder=6)

    # Key stats row
    stats = [
        ('Training Set', '~41,000\nsamples'),
        ('Validation Set', '1,000\nsamples'),
        ('Learning Rate', 'α = 0.10'),
        ('Iterations', '500'),
        ('Val Accuracy', '80.1%'),
    ]
    for i, (label, val) in enumerate(stats):
        x0 = 0.05 + i * 0.19
        box = FancyBboxPatch((x0, 0.14), 0.16, 0.13,
                             boxstyle='round,pad=0.01',
                             facecolor=hex_rgba(C_DARK, 0.0),
                             edgecolor=hex_rgba(C_YELLOW, 0.8), linewidth=1.5, zorder=4)
        ax.add_patch(box)
        ax.text(x0 + 0.08, 0.235, label,
                ha='center', va='center', fontsize=8, color=C_YELLOW, zorder=5)
        ax.text(x0 + 0.08, 0.175, val,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=C_WHITE, linespacing=1.4, zorder=5)

    ax.text(0.5, 0.05, 'Built with NumPy  •  March 2026',
            ha='center', va='center', fontsize=10, color=hex_rgba(C_DGRAY, 0.9), zorder=5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def page_dataset(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_LGRAY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_facecolor(C_LGRAY)

    # Header
    header = plt.Rectangle((0, 7.7), 11, 0.8, color=C_DARK, zorder=2)
    ax.add_patch(header)
    ax.text(0.35, 8.1, '2', ha='center', va='center', fontsize=18,
            fontweight='bold', color=C_INPUT, zorder=3)
    ax.text(5.5, 8.1, 'Dataset & Preprocessing', ha='center', va='center',
            fontsize=20, fontweight='bold', color=C_WHITE, zorder=3)

    # Dataset description
    ax.text(0.4, 7.45, 'MNIST Dataset', fontsize=14, fontweight='bold',
            color=C_DARK, va='top')
    desc = (
        "The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels.\n"
        "Each pixel value ranges 0–255. The dataset is split into a training set (~41,000 samples),\n"
        "a cross-validation set (1,000 samples), and a test set for final evaluation.\n"
        "Pixel values are normalized to [0, 1] by dividing by 255."
    )
    ax.text(0.4, 7.20, desc, fontsize=9.5, color=C_DARK, va='top', linespacing=1.6)

    # Code snippet — loading
    code_bg = FancyBboxPatch((0.35, 5.55), 4.8, 1.45,
                             boxstyle='round,pad=0.05',
                             facecolor='#1E2732', edgecolor='#34495E', linewidth=1.5, zorder=3)
    ax.add_patch(code_bg)
    ax.text(0.55, 6.88, '# Load & preprocess data', fontsize=8.5,
            color='#7F8C8D', family='monospace', va='top', zorder=4)
    code_lines = [
        ("data = pd.read_csv('train.csv')", '#ECF0F1'),
        ("data = np.array(data)", '#ECF0F1'),
        ("np.random.shuffle(data)", '#ECF0F1'),
        ("", '#ECF0F1'),
        ("X_cv  = data[:1000, 1:].T  / 255.", '#3498DB'),
        ("Y_cv  = data[:1000, 0]", '#3498DB'),
        ("X_train = data[1000:, 1:].T / 255.", '#2ECC71'),
        ("Y_train = data[1000:, 0]", '#2ECC71'),
    ]
    for j, (line, col) in enumerate(code_lines):
        ax.text(0.55, 6.70 - j*0.155, line, fontsize=8, color=col,
                family='monospace', va='top', zorder=4)

    # Code snippet — shapes
    code_bg2 = FancyBboxPatch((5.85, 5.55), 4.8, 1.45,
                              boxstyle='round,pad=0.05',
                              facecolor='#1E2732', edgecolor='#34495E', linewidth=1.5, zorder=3)
    ax.add_patch(code_bg2)
    ax.text(6.05, 6.88, '# Verify shapes', fontsize=8.5,
            color='#7F8C8D', family='monospace', va='top', zorder=4)
    shape_lines = [
        ("X_train.shape  →  (784, 41000)", '#2ECC71'),
        ("Y_train.shape  →  (41000,)",     '#2ECC71'),
        ("X_cv.shape     →  (784, 1000)",  '#3498DB'),
        ("Y_cv.shape     →  (1000,)",      '#3498DB'),
        ("", '#ECF0F1'),
        ("# Each column = one image", '#7F8C8D'),
        ("# Each row    = one pixel feature", '#7F8C8D'),
    ]
    for j, (line, col) in enumerate(shape_lines):
        ax.text(6.05, 6.70 - j*0.155, line, fontsize=8, color=col,
                family='monospace', va='top', zorder=4)

    # Variable shapes table
    ax.text(0.4, 5.35, 'Variable Shape Reference', fontsize=13, fontweight='bold',
            color=C_DARK, va='top')

    headers = ['Variable', 'Shape', 'Description', 'Role']
    col_x   = [0.35, 1.85, 3.15, 7.2]
    col_w   = [1.45, 1.25, 3.95, 3.25]

    # Header row
    for k, (hdr, cx, cw) in enumerate(zip(headers, col_x, col_w)):
        hbox = plt.Rectangle((cx, 4.72), cw-0.05, 0.38, color=C_DARK, zorder=3)
        ax.add_patch(hbox)
        ax.text(cx + (cw-0.05)/2, 4.91, hdr, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=C_WHITE, zorder=4)

    rows = [
        ('X_train', '(784, ~41000)', 'Training images — columns are samples', 'Input data'),
        ('Y_train', '(~41000,)',     'Training labels 0–9',                    'Targets'),
        ('X_cv',    '(784, 1000)',   'Cross-validation images',                'Validation input'),
        ('Y_cv',    '(1000,)',       'Cross-validation labels',                'Validation targets'),
        ('W1',      '(10, 784)',     'Hidden layer weight matrix',             'Learned parameter'),
        ('b1',      '(10, 1)',       'Hidden layer bias vector',               'Learned parameter'),
        ('W2',      '(10, 10)',      'Output layer weight matrix',             'Learned parameter'),
        ('b2',      '(10, 1)',       'Output layer bias vector',               'Learned parameter'),
        ('Z1, A1',  '(10, m)',       'Hidden layer pre/post activation',       'Intermediate'),
        ('Z2, A2',  '(10, m)',       'Output layer pre/post activation',       'Intermediate'),
    ]
    row_colors = [C_INPUT, C_DARK, C_INPUT, C_DARK,
                  C_HIDDEN, C_HIDDEN, C_OUTPUT, C_OUTPUT,
                  C_WEIGHT, C_WEIGHT]
    for ri, (row_data, rc) in enumerate(zip(rows, row_colors)):
        y0 = 4.72 - (ri+1)*0.37
        bg_col = '#F8F9FA' if ri % 2 == 0 else C_WHITE
        for k, (val, cx, cw) in enumerate(zip(row_data, col_x, col_w)):
            bg = plt.Rectangle((cx, y0), cw-0.05, 0.34,
                                color=bg_col, zorder=3)
            ax.add_patch(bg)
            fc = hex_rgba(rc, 0.9) if k == 0 else C_DARK
            fw = 'bold' if k == 0 else 'normal'
            ax.text(cx + (cw-0.05)/2, y0 + 0.17, val,
                    ha='center', va='center', fontsize=8.5,
                    color=fc, fontweight=fw, zorder=4, family='monospace' if k < 2 else 'sans-serif')

    # Normalization note
    note_bg = FancyBboxPatch((0.35, 0.18), 10.3, 0.52,
                             boxstyle='round,pad=0.05',
                             facecolor=hex_rgba(C_YELLOW, 0.15),
                             edgecolor=hex_rgba(C_YELLOW, 0.8), linewidth=1.5, zorder=3)
    ax.add_patch(note_bg)
    ax.text(0.6, 0.46, 'Normalization:', fontsize=9.5, fontweight='bold',
            color=C_DARK, va='center', zorder=4)
    ax.text(1.95, 0.46,
            'Pixel values are divided by 255 to scale inputs to [0, 1]. '
            'This improves gradient stability and speeds up convergence during training.',
            fontsize=9, color=C_DARK, va='center', zorder=4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
def page_architecture(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#F0F4F8')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_facecolor('#F0F4F8')

    # Header
    header = plt.Rectangle((0, 7.7), 11, 0.8, color=C_DARK, zorder=2)
    ax.add_patch(header)
    ax.text(0.35, 8.1, '3', ha='center', va='center', fontsize=18,
            fontweight='bold', color=C_HIDDEN, zorder=3)
    ax.text(5.5, 8.1, 'Neural Network Architecture', ha='center', va='center',
            fontsize=20, fontweight='bold', color=C_WHITE, zorder=3)

    # Layer x-positions
    x_in  = 1.6
    x_hid = 5.5
    x_out = 9.4

    # Input nodes — show x1, x2, x3, ..., x784
    in_labels  = ['x₁', 'x₂', 'x₃', '⋮', 'x₇₈₄']
    in_y_slots = [6.7, 6.0, 5.3, 4.55, 3.8]

    # Hidden nodes — h1..h10 (show all 10)
    hid_labels = [f'h{i}' for i in range(1, 11)]
    hid_y = np.linspace(7.0, 1.2, 10)

    # Output nodes — digits 0..9
    out_labels = [str(i) for i in range(10)]
    out_y = np.linspace(7.0, 1.2, 10)

    node_r = 0.28

    # Draw connections (input→hidden) — sparse subset for clarity
    real_in_ys  = [6.7, 6.0, 5.3, 3.8]   # skip ellipsis
    for iy in real_in_ys:
        for hy in hid_y:
            ax.plot([x_in + node_r, x_hid - node_r], [iy, hy],
                    color=hex_rgba(C_INPUT, 0.08), lw=0.5, zorder=1)

    # Draw connections (hidden→output)
    for hy in hid_y:
        for oy in out_y:
            ax.plot([x_hid + node_r, x_out - node_r], [hy, oy],
                    color=hex_rgba(C_OUTPUT, 0.12), lw=0.5, zorder=1)

    # Draw INPUT nodes
    for lbl, iy in zip(in_labels, in_y_slots):
        if lbl == '⋮':
            ax.text(x_in, iy, '⋮', ha='center', va='center',
                    fontsize=20, color=C_INPUT, zorder=5)
        else:
            c = plt.Circle((x_in, iy), node_r, color=C_INPUT, zorder=4)
            ax.add_patch(c)
            ax.text(x_in, iy, lbl, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=C_WHITE, zorder=5)

    # Draw HIDDEN nodes
    for lbl, hy in zip(hid_labels, hid_y):
        c = plt.Circle((x_hid, hy), node_r, color=C_HIDDEN, zorder=4)
        ax.add_patch(c)
        ax.text(x_hid, hy, lbl, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=C_WHITE, zorder=5)

    # Draw OUTPUT nodes
    for lbl, oy in zip(out_labels, out_y):
        c = plt.Circle((x_out, oy), node_r, color=C_OUTPUT, zorder=4)
        ax.add_patch(c)
        ax.text(x_out, oy, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=C_WHITE, zorder=5)

    # Layer labels A[0], A[1], A[2]
    for x, lbl, col in [(x_in, 'A[0] = X', C_INPUT),
                         (x_hid, 'A[1]',   C_HIDDEN),
                         (x_out, 'A[2]',   C_OUTPUT)]:
        ax.text(x, 7.45, lbl, ha='center', va='center', fontsize=12,
                fontweight='bold', color=col, zorder=5)

    # Unit counts
    for x, cnt, col in [(x_in, '784 units', C_INPUT),
                         (x_hid, '10 units', C_HIDDEN),
                         (x_out, '10 units', C_OUTPUT)]:
        ax.text(x, 0.72, cnt, ha='center', va='center', fontsize=9,
                color=col, fontstyle='italic', zorder=5)

    # Activation labels
    for x, act, col in [(x_in, 'Input', C_INPUT),
                         (x_hid, 'ReLU', C_HIDDEN),
                         (x_out, 'Softmax', C_OUTPUT)]:
        ab = FancyBboxPatch((x - 0.55, 0.32), 1.1, 0.28,
                            boxstyle='round,pad=0.04',
                            facecolor=hex_rgba(col, 0.2),
                            edgecolor=hex_rgba(col, 0.9), linewidth=1.5, zorder=4)
        ax.add_patch(ab)
        ax.text(x, 0.46, act, ha='center', va='center', fontsize=9,
                fontweight='bold', color=col, zorder=5)

    # Weight/bias labels between layers
    # W1, b1
    mid1 = (x_in + x_hid) / 2
    wb1_box = FancyBboxPatch((mid1 - 0.8, 5.5), 1.6, 0.7,
                             boxstyle='round,pad=0.06',
                             facecolor=hex_rgba(C_WEIGHT, 0.15),
                             edgecolor=hex_rgba(C_WEIGHT, 0.9), linewidth=2, zorder=6)
    ax.add_patch(wb1_box)
    ax.text(mid1, 5.97, 'W1 (10×784)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_WEIGHT, zorder=7)
    ax.text(mid1, 5.64, 'b1 (10×1)', ha='center', va='center',
            fontsize=9, color=C_WEIGHT, zorder=7)

    # Z1 formula
    z1_box = FancyBboxPatch((mid1 - 0.9, 4.45), 1.8, 0.55,
                            boxstyle='round,pad=0.06',
                            facecolor=hex_rgba(C_INPUT, 0.12),
                            edgecolor=hex_rgba(C_INPUT, 0.7), linewidth=1.5, zorder=6)
    ax.add_patch(z1_box)
    ax.text(mid1, 4.72, 'Z1 = W1·X + b1', ha='center', va='center',
            fontsize=9, color=C_INPUT, fontweight='bold', zorder=7)

    # W2, b2
    mid2 = (x_hid + x_out) / 2
    wb2_box = FancyBboxPatch((mid2 - 0.75, 5.5), 1.5, 0.7,
                             boxstyle='round,pad=0.06',
                             facecolor=hex_rgba(C_WEIGHT, 0.15),
                             edgecolor=hex_rgba(C_WEIGHT, 0.9), linewidth=2, zorder=6)
    ax.add_patch(wb2_box)
    ax.text(mid2, 5.97, 'W2 (10×10)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_WEIGHT, zorder=7)
    ax.text(mid2, 5.64, 'b2 (10×1)', ha='center', va='center',
            fontsize=9, color=C_WEIGHT, zorder=7)

    # Z2 formula
    z2_box = FancyBboxPatch((mid2 - 0.9, 4.45), 1.8, 0.55,
                            boxstyle='round,pad=0.06',
                            facecolor=hex_rgba(C_HIDDEN, 0.12),
                            edgecolor=hex_rgba(C_HIDDEN, 0.7), linewidth=1.5, zorder=6)
    ax.add_patch(z2_box)
    ax.text(mid2, 4.72, 'Z2 = W2·A1 + b2', ha='center', va='center',
            fontsize=9, color=C_HIDDEN, fontweight='bold', zorder=7)

    # Shape annotation boxes
    for x, shape, col in [(x_in, '(784, m)', C_INPUT),
                           (x_hid, '(10, m)', C_HIDDEN),
                           (x_out, '(10, m)', C_OUTPUT)]:
        sb = FancyBboxPatch((x - 0.55, 1.05), 1.1, 0.28,
                            boxstyle='round,pad=0.03',
                            facecolor='white',
                            edgecolor=hex_rgba(col, 0.6), linewidth=1, zorder=4)
        ax.add_patch(sb)
        ax.text(x, 1.19, shape, ha='center', va='center', fontsize=8,
                color=col, family='monospace', zorder=5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Forward Propagation
# ══════════════════════════════════════════════════════════════════════════════
def page_forward(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_LGRAY)

    # Header axis
    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor(C_DARK); ax_hdr.axis('off')
    ax_hdr.text(0.03, 0.5, '4', ha='center', va='center', fontsize=18,
                fontweight='bold', color=C_OUTPUT, transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.5, 'Forward Propagation', ha='center', va='center',
                fontsize=20, fontweight='bold', color=C_WHITE, transform=ax_hdr.transAxes)

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.85])
    ax.set_facecolor(C_LGRAY); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)

    steps = [
        ('Step 1', 'Z1 = W1 · X + b1',
         'Compute the pre-activation of the hidden layer.\nW1 is (10,784), X is (784,m) → Z1 is (10,m)',
         C_INPUT, 6.7),
        ('Step 2', 'A1 = ReLU(Z1) = max(0, Z1)',
         'Apply ReLU activation element-wise.\nNegative values → 0; positive values unchanged. A1 is (10,m)',
         C_HIDDEN, 5.3),
        ('Step 3', 'Z2 = W2 · A1 + b2',
         'Compute the pre-activation of the output layer.\nW2 is (10,10), A1 is (10,m) → Z2 is (10,m)',
         C_OUTPUT, 3.9),
        ('Step 4', 'A2 = softmax(Z2)',
         'Apply Softmax: exp(Z2) / Σ exp(Z2) column-wise.\nA2[i,j] = probability that sample j is digit i.',
         C_WEIGHT, 2.5),
    ]

    for step, formula, desc, col, y_top in steps:
        # Colored left bar
        bar = plt.Rectangle((0.05, y_top - 0.9), 0.07, 0.9,
                             color=col, zorder=3)
        ax.add_patch(bar)
        # Box
        box = FancyBboxPatch((0.12, y_top - 0.92), 9.7, 0.94,
                             boxstyle='round,pad=0.04',
                             facecolor='white',
                             edgecolor=hex_rgba(col, 0.4), linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(0.32, y_top - 0.15, step, fontsize=9.5, fontweight='bold',
                color=col, va='top', zorder=4)
        ax.text(2.5, y_top - 0.15, formula, fontsize=12, fontweight='bold',
                color=C_DARK, va='top', family='monospace', zorder=4)
        ax.text(0.32, y_top - 0.52, desc, fontsize=8.5, color=C_DGRAY,
                va='top', linespacing=1.5, zorder=4)

    # Mini plots
    ax_relu = fig.add_axes([0.06, 0.07, 0.28, 0.22])
    z = np.linspace(-3, 3, 200)
    ax_relu.plot(z, np.maximum(0, z), color=C_HIDDEN, lw=2.5, label='ReLU(z)')
    ax_relu.plot(z, z, color=C_DGRAY, lw=1, linestyle='--', alpha=0.5, label='y = z')
    ax_relu.axhline(0, color='black', lw=0.5); ax_relu.axvline(0, color='black', lw=0.5)
    ax_relu.set_title('ReLU Activation', fontsize=10, fontweight='bold', color=C_HIDDEN)
    ax_relu.legend(fontsize=7); ax_relu.set_facecolor('white')
    ax_relu.tick_params(labelsize=7)
    ax_relu.fill_between(z, 0, np.maximum(0, z), alpha=0.15, color=C_HIDDEN)

    ax_soft = fig.add_axes([0.38, 0.07, 0.28, 0.22])
    z_ex = np.array([2.0, 1.0, 0.1, -0.5, 3.0, -1.0, 0.5, 1.5, -0.2, 0.8])
    s_ex = np.exp(z_ex) / np.sum(np.exp(z_ex))
    bars = ax_soft.bar(range(10), s_ex, color=[C_WEIGHT]*10)
    bars[4].set_color(C_RED)
    ax_soft.set_title('Softmax Output\n(example)', fontsize=10, fontweight='bold', color=C_WEIGHT)
    ax_soft.set_xlabel('Digit class', fontsize=8)
    ax_soft.set_ylabel('Probability', fontsize=8)
    ax_soft.set_facecolor('white'); ax_soft.tick_params(labelsize=7)
    ax_soft.set_xticks(range(10))

    # Shape reference
    ax_shapes = fig.add_axes([0.70, 0.07, 0.27, 0.22])
    ax_shapes.axis('off')
    ax_shapes.set_facecolor('white')
    ax_shapes.set_title('Variable Shapes', fontsize=10, fontweight='bold', color=C_DARK)
    shape_rows = [
        ('X', '(784, m)', C_INPUT),
        ('W1', '(10, 784)', C_INPUT),
        ('Z1 = A1', '(10, m)', C_HIDDEN),
        ('W2', '(10, 10)', C_OUTPUT),
        ('Z2 = A2', '(10, m)', C_WEIGHT),
    ]
    for si, (var, shp, col) in enumerate(shape_rows):
        y_ = 0.82 - si * 0.17
        ax_shapes.text(0.05, y_, var, transform=ax_shapes.transAxes,
                       fontsize=9, fontweight='bold', color=col, va='center', family='monospace')
        ax_shapes.text(0.45, y_, shp, transform=ax_shapes.transAxes,
                       fontsize=9, color=C_DARK, va='center', family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Backward Propagation & Parameter Updates
# ══════════════════════════════════════════════════════════════════════════════
def page_backward(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_LGRAY)

    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor(C_DARK); ax_hdr.axis('off')
    ax_hdr.text(0.03, 0.5, '5', ha='center', va='center', fontsize=18,
                fontweight='bold', color=C_RED, transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.5, 'Backward Propagation & Parameter Updates',
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=C_WHITE, transform=ax_hdr.transAxes)

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.85])
    ax.set_facecolor(C_LGRAY); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)

    # Section: Output layer gradients
    ax.text(0.1, 7.35, 'Output Layer Gradients', fontsize=12, fontweight='bold',
            color=C_WEIGHT, va='top')
    out_grads = [
        ('dZ2', 'dZ2 = A2 - Y_one_hot',       '(10, m)', 'Error at output (predicted − true one-hot)'),
        ('dW2', 'dW2 = (1/m) · dZ2 · A1.T',   '(10, 10)', 'Gradient of loss w.r.t. W2'),
        ('db2', 'db2 = (1/m) · Σ dZ2',         '(10, 1)', 'Gradient of loss w.r.t. b2'),
    ]
    for gi, (var, formula, shape, desc) in enumerate(out_grads):
        y0 = 6.85 - gi * 0.78
        box = FancyBboxPatch((0.05, y0 - 0.62), 9.8, 0.65,
                             boxstyle='round,pad=0.04', facecolor='white',
                             edgecolor=hex_rgba(C_WEIGHT, 0.5), linewidth=1.5, zorder=2)
        ax.add_patch(box)
        bar = plt.Rectangle((0.05, y0 - 0.62), 0.08, 0.65, color=C_WEIGHT, zorder=3)
        ax.add_patch(bar)
        ax.text(0.25, y0 - 0.08, var, fontsize=10, fontweight='bold',
                color=C_WEIGHT, va='top', zorder=4)
        ax.text(0.7, y0 - 0.08, formula, fontsize=10.5, color=C_DARK,
                va='top', family='monospace', fontweight='bold', zorder=4)
        ax.text(7.5, y0 - 0.08, shape, fontsize=8.5, color=C_DGRAY,
                va='top', family='monospace', zorder=4)
        ax.text(0.25, y0 - 0.42, desc, fontsize=8.5, color=C_DGRAY,
                va='top', zorder=4)

    # Section: Hidden layer gradients
    ax.text(0.1, 4.6, 'Hidden Layer Gradients', fontsize=12, fontweight='bold',
            color=C_HIDDEN, va='top')
    hid_grads = [
        ('dZ1', 'dZ1 = W2.T · dZ2  *  ReLU_deriv(Z1)', '(10, m)', 'Back-propagated error through ReLU'),
        ('dW1', 'dW1 = (1/m) · dZ1 · X.T',              '(10, 784)', 'Gradient of loss w.r.t. W1'),
        ('db1', 'db1 = (1/m) · Σ dZ1',                  '(10, 1)', 'Gradient of loss w.r.t. b1'),
    ]
    for gi, (var, formula, shape, desc) in enumerate(hid_grads):
        y0 = 4.15 - gi * 0.78
        box = FancyBboxPatch((0.05, y0 - 0.62), 9.8, 0.65,
                             boxstyle='round,pad=0.04', facecolor='white',
                             edgecolor=hex_rgba(C_HIDDEN, 0.5), linewidth=1.5, zorder=2)
        ax.add_patch(box)
        bar = plt.Rectangle((0.05, y0 - 0.62), 0.08, 0.65, color=C_HIDDEN, zorder=3)
        ax.add_patch(bar)
        ax.text(0.25, y0 - 0.08, var, fontsize=10, fontweight='bold',
                color=C_HIDDEN, va='top', zorder=4)
        ax.text(0.7, y0 - 0.08, formula, fontsize=10, color=C_DARK,
                va='top', family='monospace', fontweight='bold', zorder=4)
        ax.text(7.8, y0 - 0.08, shape, fontsize=8.5, color=C_DGRAY,
                va='top', family='monospace', zorder=4)
        ax.text(0.25, y0 - 0.42, desc, fontsize=8.5, color=C_DGRAY,
                va='top', zorder=4)

    # Parameter update formulas
    ax.text(0.1, 1.75, 'Parameter Update (Gradient Descent,  α = 0.10)', fontsize=12,
            fontweight='bold', color=C_RED, va='top')
    upd_box = FancyBboxPatch((0.05, 0.5), 9.8, 1.1,
                             boxstyle='round,pad=0.06', facecolor='white',
                             edgecolor=hex_rgba(C_RED, 0.6), linewidth=2, zorder=2)
    ax.add_patch(upd_box)
    update_lines = [
        'W1 ← W1 − α · dW1',
        'b1 ← b1 − α · db1',
        'W2 ← W2 − α · dW2',
        'b2 ← b2 − α · db2',
    ]
    for ui, uline in enumerate(update_lines):
        ax.text(0.4 + ui * 2.4, 1.08, uline, ha='left', va='center',
                fontsize=10, fontweight='bold', color=C_DARK,
                family='monospace', zorder=4)

    # One-hot note
    note = FancyBboxPatch((0.05, 0.08), 9.8, 0.35,
                          boxstyle='round,pad=0.04',
                          facecolor=hex_rgba(C_YELLOW, 0.15),
                          edgecolor=hex_rgba(C_YELLOW, 0.8), linewidth=1.5, zorder=3)
    ax.add_patch(note)
    ax.text(0.25, 0.26, 'one_hot(Y):',
            fontsize=9, fontweight='bold', color=C_DARK, va='center', zorder=4)
    ax.text(1.5, 0.26,
            'Converts label vector (m,) → matrix (10, m). '
            'Column j has a 1 in row Y[j] and 0s elsewhere. '
            'ReLU_deriv(Z1) = (Z1 > 0) — returns 1 where active, 0 where not.',
            fontsize=8.5, color=C_DARK, va='center', zorder=4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Code Documentation
# ══════════════════════════════════════════════════════════════════════════════
def page_code(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_LGRAY)

    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor(C_DARK); ax_hdr.axis('off')
    ax_hdr.text(0.03, 0.5, '6', ha='center', va='center', fontsize=18,
                fontweight='bold', color=C_TEAL, transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.5, 'Code Documentation — Function Reference',
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=C_WHITE, transform=ax_hdr.transAxes)

    ax = fig.add_axes([0.03, 0.02, 0.94, 0.87])
    ax.set_facecolor(C_LGRAY); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6)

    functions = [
        ('init_params()',
         'Initialise all weight matrices and bias vectors with uniform random values in [−0.5, 0.5].',
         'None',
         'W1 (10,784), b1 (10,1), W2 (10,10), b2 (10,1)',
         C_INPUT),
        ('ReLU(Z)',
         'Element-wise rectified linear unit: ReLU(z) = max(0, z).',
         'Z — array of any shape',
         'Array of same shape with negatives zeroed',
         C_HIDDEN),
        ('softmax(Z)',
         'Column-wise softmax. Converts raw scores to a probability distribution over 10 classes.',
         'Z — (10, m) pre-activation matrix',
         '(10, m) probability matrix; each column sums to 1',
         C_OUTPUT),
        ('forward_prop(W1,b1,W2,b2,X)',
         'Run a full forward pass through the network.',
         'W1,b1,W2,b2 — params; X — (784,m) input',
         'Z1, A1, Z2, A2 — all (10,m)',
         C_WEIGHT),
        ('ReLU_deriv(Z)',
         'Derivative of ReLU: returns 1.0 where Z > 0, else 0.0.',
         'Z — (10, m) pre-activation',
         'Boolean/float mask (10, m)',
         C_INPUT),
        ('one_hot(Y)',
         'Encode integer label array as a one-hot matrix.',
         'Y — integer array (m,) with values 0–9',
         '(10, m) one-hot matrix',
         C_HIDDEN),
        ('backward_prop(Z1,A1,Z2,A2,W1,W2,X,Y)',
         'Compute all gradients via backpropagation.',
         'All forward-pass outputs plus W1, W2, X, Y',
         'dW1, db1, dW2, db2',
         C_OUTPUT),
        ('update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)',
         'Apply one gradient descent step with learning rate alpha.',
         'Current params + gradients + alpha=0.10',
         'Updated W1, b1, W2, b2',
         C_WEIGHT),
        ('get_predictions(A2)',
         'Choose the class with the highest probability for each sample.',
         'A2 — (10, m) softmax output',
         'Integer array (m,) of predicted labels',
         C_INPUT),
        ('get_accuracy(predictions, Y)',
         'Fraction of correctly classified samples.',
         'predictions (m,), Y — true labels (m,)',
         'Float in [0, 1]',
         C_HIDDEN),
        ('gradient_descent(X,Y,alpha,iterations)',
         'Main training loop: init → forward → backward → update, repeated for `iterations` steps.',
         'X,Y — data; alpha=0.10; iterations=500',
         'Final W1, b1, W2, b2',
         C_OUTPUT),
        ('make_predictions(X,W1,b1,W2,b2)',
         'Run forward prop and return predicted digit classes.',
         'X — (784,m); trained params',
         'Integer array (m,) of predictions',
         C_WEIGHT),
    ]

    n = len(functions)
    row_h = 7.4 / n
    for i, (sig, desc, params, ret, col) in enumerate(functions):
        y0 = 7.4 - (i + 1) * row_h
        bg = '#F8F9FA' if i % 2 == 0 else C_WHITE
        box = plt.Rectangle((0, y0), 10, row_h - 0.04, color=bg, zorder=2)
        ax.add_patch(box)
        bar = plt.Rectangle((0, y0), 0.1, row_h - 0.04, color=col, zorder=3)
        ax.add_patch(bar)
        yc = y0 + (row_h - 0.04) / 2
        ax.text(0.25, yc + 0.09, sig, fontsize=8.2, fontweight='bold',
                color=col, va='center', family='monospace', zorder=4)
        ax.text(0.25, yc - 0.09, desc, fontsize=7.8, color=C_DARK,
                va='center', zorder=4)
        ax.text(5.2, yc + 0.09, 'In: ' + params, fontsize=7.2, color=C_DGRAY,
                va='center', zorder=4)
        ax.text(5.2, yc - 0.09, 'Out: ' + ret, fontsize=7.2, color=C_DARK,
                va='center', zorder=4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Training Results
# ══════════════════════════════════════════════════════════════════════════════
def page_results(pdf):
    acc_vals = [
        0.1187, 0.1730, 0.2647, 0.3310, 0.3740, 0.4079, 0.4353, 0.4586,
        0.4815, 0.5041, 0.5261, 0.5472, 0.5666, 0.5840, 0.6014, 0.6178,
        0.6327, 0.6460, 0.6593, 0.6714, 0.6824, 0.6929, 0.7031, 0.7115,
        0.7184, 0.7265, 0.7338, 0.7395, 0.7454, 0.7511, 0.7558, 0.7605,
        0.7653, 0.7688, 0.7724, 0.7761, 0.7794, 0.7829, 0.7852, 0.7883,
        0.7911, 0.7937, 0.7962, 0.7985, 0.8002, 0.8023, 0.8044, 0.8065,
        0.8086, 0.8105,
    ]
    iters = [i * 10 for i in range(1, len(acc_vals) + 1)]

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(C_LGRAY)

    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor(C_DARK); ax_hdr.axis('off')
    ax_hdr.text(0.03, 0.5, '7', ha='center', va='center', fontsize=18,
                fontweight='bold', color=C_YELLOW, transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.5, 'Training Results', ha='center', va='center',
                fontsize=20, fontweight='bold', color=C_WHITE, transform=ax_hdr.transAxes)

    # Main accuracy plot
    ax_plot = fig.add_axes([0.06, 0.35, 0.60, 0.52])
    ax_plot.plot(iters, acc_vals, color=C_INPUT, lw=2.5, marker='o',
                 markersize=3.5, label='Training Accuracy', zorder=4)
    ax_plot.axhline(0.801, color=C_RED, lw=2, linestyle='--',
                    label='Validation Accuracy (80.1%)', zorder=3)

    # Shade area
    ax_plot.fill_between(iters, acc_vals, alpha=0.15, color=C_INPUT)

    # Milestone annotations
    milestones = [(10, 0.1187, '11.9%\nStart'), (100, 0.5041, '50.4%'),
                  (250, 0.7338, '73.4%'), (500, 0.8105, '81.1%\nFinal')]
    for it, ac, lbl in milestones:
        ax_plot.annotate(lbl, xy=(it, ac),
                         xytext=(it + 15, ac + 0.055),
                         fontsize=7.5, color=C_DARK,
                         arrowprops=dict(arrowstyle='->', color=C_DGRAY, lw=1),
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor=C_INPUT, linewidth=1))

    ax_plot.set_xlabel('Iteration', fontsize=10)
    ax_plot.set_ylabel('Accuracy', fontsize=10)
    ax_plot.set_title('Training Accuracy over 500 Iterations', fontsize=11,
                      fontweight='bold', color=C_DARK)
    ax_plot.set_ylim(0, 1.0); ax_plot.set_xlim(0, 510)
    ax_plot.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_plot.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax_plot.legend(fontsize=8.5, loc='lower right')
    ax_plot.grid(True, alpha=0.3); ax_plot.set_facecolor('white')
    ax_plot.tick_params(labelsize=8.5)

    # Stats boxes (right column)
    stats = [
        ('Initial Acc.', '11.9%', C_RED),
        ('Final Train Acc.', '81.1%', C_INPUT),
        ('Val Accuracy', '80.1%', C_OUTPUT),
        ('Learning Rate', 'α = 0.10', C_WEIGHT),
        ('Iterations', '500', C_HIDDEN),
        ('Log Interval', 'Every 10', C_DGRAY),
    ]
    for si, (label, val, col) in enumerate(stats):
        y0 = 0.88 - si * 0.095
        ax_s = fig.add_axes([0.71, y0 - 0.06, 0.26, 0.08])
        ax_s.set_facecolor('white')
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis('off')
        bar_r = plt.Rectangle((0, 0), 0.07, 1, color=col, transform=ax_s.transAxes)
        ax_s.add_patch(bar_r)
        ax_s.text(0.12, 0.72, label, transform=ax_s.transAxes,
                  fontsize=8, color=C_DGRAY, va='center')
        ax_s.text(0.12, 0.28, val, transform=ax_s.transAxes,
                  fontsize=11, fontweight='bold', color=col, va='center')

    # Training loop flow diagram
    ax_flow = fig.add_axes([0.04, 0.05, 0.92, 0.26])
    ax_flow.set_xlim(0, 10); ax_flow.set_ylim(0, 2.2)
    ax_flow.axis('off')
    ax_flow.set_facecolor('#F8F9FA')

    ax_flow.text(5, 2.08, 'Training Loop  (gradient_descent)', ha='center',
                 fontsize=10, fontweight='bold', color=C_DARK, va='center')

    flow_steps = [
        ('init_params()', C_INPUT, 0.5),
        ('forward_prop()', C_HIDDEN, 2.4),
        ('get_accuracy()', C_TEAL, 4.3),
        ('backward_prop()', C_OUTPUT, 6.2),
        ('update_params()', C_WEIGHT, 8.1),
    ]
    for lbl, col, x0 in flow_steps:
        fbox = FancyBboxPatch((x0, 0.55), 1.6, 0.8,
                              boxstyle='round,pad=0.08',
                              facecolor=hex_rgba(col, 0.2),
                              edgecolor=hex_rgba(col, 0.9), linewidth=1.5, zorder=3)
        ax_flow.add_patch(fbox)
        ax_flow.text(x0 + 0.8, 0.95, lbl, ha='center', va='center',
                     fontsize=8, fontweight='bold', color=col, zorder=4)

    # Arrows between flow boxes
    arrow_xs = [(2.1, 2.4), (4.0, 4.3), (5.9, 6.2), (7.8, 8.1)]
    for x1, x2 in arrow_xs:
        ax_flow.annotate('', xy=(x2, 0.95), xytext=(x1, 0.95),
                         arrowprops=dict(arrowstyle='->', color=C_DGRAY, lw=1.5))

    # Loop-back arrow
    ax_flow.annotate('', xy=(0.5, 0.55), xytext=(9.7, 0.55),
                     xycoords='data', textcoords='data',
                     arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5,
                                     connectionstyle='arc3,rad=-0.35'))
    ax_flow.text(5.1, 0.12, '← repeat for 500 iterations →', ha='center',
                 fontsize=8, color=C_RED, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with PdfPages(OUTPUT_PATH) as pdf:
        print('Generating Page 1: Title Page...')
        page_title(pdf)
        print('Generating Page 2: Dataset & Preprocessing...')
        page_dataset(pdf)
        print('Generating Page 3: Architecture Diagram...')
        page_architecture(pdf)
        print('Generating Page 4: Forward Propagation...')
        page_forward(pdf)
        print('Generating Page 5: Backward Propagation...')
        page_backward(pdf)
        print('Generating Page 6: Code Documentation...')
        page_code(pdf)
        print('Generating Page 7: Training Results...')
        page_results(pdf)

        d = pdf.infodict()
        d['Title']   = 'Neural Network for MNIST Digit Classification'
        d['Author']  = 'Generated by matplotlib'
        d['Subject'] = 'Deep Learning Documentation'
        d['Keywords'] = 'neural network, MNIST, NumPy, backpropagation'

    print(f'\nPDF saved to: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
