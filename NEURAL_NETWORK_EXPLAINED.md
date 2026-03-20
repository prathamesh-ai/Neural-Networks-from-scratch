# Neural Network from Scratch — The Complete Workflow

> A step-by-step explanation of **why** each component exists, not just what it does.

---

## The Big Picture

A neural network is a function that maps inputs to outputs by learning from examples.
The keyword is **learning** — we don't program the rules manually. We let the network discover them by repeatedly making predictions, measuring how wrong they are, and adjusting.

Every component in a neural network exists to solve one specific problem in that process.

```
Raw Data
   │
   ▼
[Input Layer]  ──→  pixels, numbers, features — just the data as-is
   │
   ▼
[Hidden Layer]  ──→  Z = W·X + b  then  A = ReLU(Z)
   │
   ▼
[Output Layer]  ──→  Z = W·A + b  then  A = Softmax(Z)
   │
   ▼
Probabilities  ──→  [0.02, 0.01, 0.03, 0.87, ...]  (sums to 1)
   │
   ▼
[Loss Function]  ──→  one number: "how wrong are we?"
   │
   ▼
[Backpropagation]  ──→  "which weights caused the error, and by how much?"
   │
   ▼
[Gradient Descent]  ──→  nudge every weight slightly in the right direction
   │
   ▼
repeat 500× → network improves from 10% → 85%+ accuracy
```

---

## Component 1 — The Neuron

### What it does
A single neuron takes `n` inputs, multiplies each by a weight, sums them up, and adds a bias:

```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
```

In vector form — the weights `W` and input `X` are 1D vectors, `b` is a scalar:

```
       W (1 × n_inputs)            X (n_inputs × 1)
  ┌─────────────────────────┐    ┌──────┐
  │  w₁   w₂   w₃  ...  wₙ │  · │  x₁  │  +  b  =  z  (scalar)
  └─────────────────────────┘    │  x₂  │
                                 │  x₃  │
                                 │  ... │
                                 │  xₙ  │
                                 └──────┘
```

**Example** with 3 inputs:
```
W = [ 0.3   -0.5   0.8 ]       X = [ 0.5 ]       b = 0.1
                                    [ 0.2 ]
                                    [ 0.8 ]

z = (0.3×0.5) + (-0.5×0.2) + (0.8×0.8) + 0.1
  =  0.15    +  -0.10      +  0.64     + 0.1
  =  0.79
```

### Why it's necessary
The weights `w` and bias `b` are the things the network *learns*. Without them there is nothing to adjust, nothing to train, nothing to learn.

The bias `b` lets a neuron activate even when all inputs are zero — it gives the neuron its own "default opinion" independent of the data.

### What would break without it
Without weights, the network is a fixed transformation — the same for every problem, with no ability to adapt.

---

## Component 2 — The Layer (Matrix Math)

### What it does
A layer is `n_neurons` neurons all reading the same input simultaneously.
Instead of computing each neuron one by one, we pack all their weights into a matrix and do one multiply:

```
Z = W · X + b
```

For a hidden layer with **128 neurons**, **784 inputs**, **m samples**:

```
  W  (128 × 784)          X  (784 × m)             b (128 × 1)     Z (128 × m)
  [actual shape shown]    [actual shape shown]
  ┌──────────────────┐    ┌──────────────┐          ┌────┐          ┌──────────┐
  │ w₁₁  w₁₂  ...    │    │ x₁₁  x₁₂ …   │          │ b₁ │          │ z₁₁  z₁₂ │
  │ w₂₁  w₂₂  ...    │  · │ x₂₁  x₂₂ …   │    +     │ b₂ │    =     │ z₂₁  z₂₂ │
  │ ...  ...   ...   │    │ ...  ...  …  │          │ .. │          │ ...  ... |
  │w₁₂₈₁ ...  ...    │    │ x₇₈₄₁ ...…   │          │b₁₂₈│          │z₁₂₈₁ ... │
  └──────────────────┘    └──────────────┘          └────┘          └──────────┘
  ↑ row i = weights          ↑ col j = one               ↑ broadcast  ↑ cell (i,j) =
    of neuron i                sample's pixels              to all m     neuron i's output
                                                            samples      for sample j
```

**Small concrete example** — 3 neurons, 4 inputs, 2 samples:

```
  W (3 × 4)                X (4 × 2)          b (3 × 1)     Z (3 × 2)
  ┌                     ┐   ┌         ┐         ┌     ┐       ┌         ┐
  │  0.3  -0.5  0.8  0.1│   │ 0.5  0.9│         │ 0.1 │       │ z₁₁  z₁₂│
  │ -0.2   0.4  0.6 -0.3│ · │ 0.2  0.1│    +    │-0.2 │   =   │ z₂₁  z₂₂│
  │  0.7  -0.1  0.2  0.5│   │ 0.8  0.3│         │ 0.3 │       │ z₃₁  z₃₂│
  └                     ┘   │ 0.4  0.7│         └     ┘       └         ┘
                            └         ┘
  ↑ 3 rows = 3 neurons      ↑ 2 cols = 2 samples             ↑ 3 neurons × 2 samples
```

Row `i` of W dotted with column `j` of X gives `Z[i,j]` — neuron `i`'s output for sample `j`.

### Why it's necessary
1. **Efficiency** — One matrix multiply computes all neurons for all samples at once. This is why neural networks run fast on GPUs — they are massively parallel matrix machines.
2. **Compositional abstraction** — Each layer computes a new representation built on top of the previous layer's outputs. In theory, deeper layers can capture more complex patterns. However, *what* each neuron actually specialises in is **not guaranteed or defined** — it depends entirely on the data, the task, and random initialisation. The common description of "Layer 1 detects edges, Layer 2 detects shapes" is an intuition observed empirically in CNNs (Convolutional Neural Networks) trained on images — it does **not** apply to plain fully-connected networks like ours, where neurons have no spatial awareness and their learned features are largely uninterpretable without explicit analysis.

### What would break without it
You'd loop over every neuron for every sample — unusably slow, and no hierarchy of features.

---

## Component 3 — ReLU Activation Function

### What it does
Applied **element-wise** to every value in Z after a hidden layer:

```
ReLU(z) = max(0, z)
```


```
  Z (3 × 2)  — before ReLU         A (3 × 2)  — after ReLU
  ┌              ┐                  ┌              ┐
  │  0.79  -0.31 │    →  ReLU  →    │  0.79   0.00 │   ← negative killed
  │ -0.44   0.55 │                  │  0.00   0.55 │   ← negative killed
  │  0.12   0.38 │                  │  0.12   0.38 │
  └              ┘                  └              ┘
  same shape, same size — only sign changes
```

### Why it's necessary

The **one and only job** of an activation function is to introduce **non-linearity**. "Capturing patterns from data" is a consequence of that — not the direct cause.

**Why non-linearity matters:** Real-world data is almost never linearly separable. In MNIST for example, the digit `0` and digit `1` cannot be separated by a straight line in pixel space — thousands of overlapping pixel configurations exist for both classes. You need a decision boundary that bends and folds through 784 dimensions. That folding is only possible because of non-linearity.

Without activations, stacking layers is mathematically useless. Two linear layers always collapse into one:

```
  Z² = W² · (W¹ · X + b¹) + b²
     = (W²W¹) · X  +  (W²b¹ + b²)
     =    W'  · X  +       b'           ← still just one linear layer
```

No matter how many layers: `depth without activation = one straight line`.

```
  Without activation          With ReLU activation
  (linear only)               (non-linear)

      ●  ○  ●                     ●  ○  ●
    ○  ●  ○                     ○  ●  ○
      ●  ○  ●                     ●  ○  ●

    Can't separate this.        Can separate this.
    Best it can do is           Boundary can bend,
    one straight line.          curve, and fold.
```

ReLU breaks linearity by treating positive and negative regions differently. Each neuron becomes a **switch**:

```
  Neuron A  (w=2, b=-1)      Neuron B  (w=-2, b=1)      Combined
  fires when x > 0.5         fires when x < 0.5         covers the full range

       ↗ active                  active ↘
  ────/──────────             ──────────\────
     /  dead zone              dead zone \
```

Their **weighted sum** can approximate any curve — this is why more neurons = more expressive network.

**The precise way to think about it:**

| Common phrasing | More precise |
|---|---|
| "Captures patterns from data" | "Learns non-linear decision boundaries" |
| "Understands the data better" | "Maps inputs to outputs not achievable with a straight line" |
| "Provides good predictions" | "Correctly classifies inputs that aren't linearly separable" |

The activation function doesn't add understanding — it adds **flexibility** to the function the network can represent.

### Why ReLU specifically
- `max(0, z)` is one operation — fast
- Derivative is 1 (active) or 0 (dead) — clean gradient, no vanishing
- Unlike Sigmoid/Tanh, it doesn't saturate — large inputs still get gradient

### What would break without it
Network collapses to linear — can only fit straight lines. MNIST accuracy would plateau ~30-40%.

---

## Component 4 — Softmax (Output Activation)

### What it does
The output layer produces raw scores called **logits** — any real number. Softmax converts them to probabilities:

```
Softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

```
  Z² (logits, 10 × 1)       A² (probabilities, 10 × 1)
  ┌       ┐                 ┌        ┐
  │  2.0  │  class 0        │  0.09  │  9%
  │  1.0  │  class 1        │  0.03  │  3%
  │  0.5  │  class 2        │  0.02  │  2%
  │ -1.0  │  class 3        │  0.00  │  0%
  │  3.5  │  class 4  →     │  0.44  │  44%  ← predicted class (argmax)
  │ -0.5  │  class 5        │  0.01  │  1%
  │  0.2  │  class 6        │  0.02  │  2%
  │  1.8  │  class 7        │  0.08  │  8%
  │ -2.0  │  class 8        │  0.00  │  0%
  │  0.9  │  class 9        │  0.03  │  3%
  └       ┘                 └        ┘
                            sum = 1.00  ✓
```

The largest logit (3.5 for class 4) wins, but all classes get *some* probability — the network expresses its full confidence distribution.

### Why it's necessary
Raw logits can't be used directly:
- They don't sum to 1 → can't interpret as probability
- They can be negative → `log(negative)` in loss function is undefined

### Why not ReLU on the output?
ReLU could zero out an entire class (set probability = 0), then `log(0) = -∞` blows up the loss.

### What would break without it
Loss computation becomes numerically undefined. Training crashes.

---

## Component 5 — Loss Function (Cross-Entropy)

### What it does
Takes the probability vector `A²` and the true label `y`, and produces a single number measuring how wrong we are:

```
L = -log( A²[y] )     ← just the probability assigned to the correct class
```

```
  A² (10 × 1)                         True label: y = 4
  ┌        ┐
  │  0.09  │  class 0
  │  0.03  │  class 1
  │  0.02  │  class 2
  │  0.00  │  class 3
  │  0.44  │  class 4  ← correct class → L = -log(0.44) = 0.82
  │  0.01  │  class 5
  │  0.02  │  class 6
  │  0.08  │  class 7
  │  0.00  │  class 8
  │  0.03  │  class 9
  └        ┘
```

Over `m` samples — one correct-class probability per sample, averaged:

```
  correct_probs  (1 × m)  — one entry per sample
  ┌──────────────────────────────────────────┐
  │  p_s1   p_s2   p_s3   p_s4  ...  p_sm   │
  └──────────────────────────────────────────┘
         ↓  apply -log to each  ↓
  ┌──────────────────────────────────────────┐
  │  L_s1   L_s2   L_s3   L_s4  ...  L_sm   │
  └──────────────────────────────────────────┘
                    ↓  average  ↓
                    L  (scalar)
```

| Predicted probability | Loss = -log(p) |
|---|---|
| `p = 0.99` (confident, correct) | `0.01` — nearly zero |
| `p = 0.50` (uncertain) | `0.69` — moderate |
| `p = 0.10` (random guess) | `2.30` — baseline |
| `p = 0.01` (confident, wrong) | `4.60` — severe penalty |

### Why it's necessary
Accuracy (0 or 1) is not differentiable — you can't compute a gradient from it. Cross-entropy is smooth everywhere, so backprop can compute precise gradients.

The `-log` curve punishes overconfident wrong answers disproportionately — forcing the network to be calibrated, not just sometimes right.

### What would break without it
No differentiable signal → no gradients → no backprop → no learning.

---

## Component 6 — Backpropagation

### What it does
Uses the chain rule to compute the gradient of the loss with respect to **every weight** in the network, walking backwards layer by layer.

```
∂L/∂W¹ = ∂L/∂Z² · ∂Z²/∂A¹ · ∂A¹/∂Z¹ · ∂Z¹/∂W¹
```

**Step-by-step for our network:**

**Step 1 — Error at output layer** (Softmax + Cross-entropy combined):
```
  dZ²  (10 × m)
  ┌                        ┐
  │ A²[0]-Y[0]  ...  ...   │
  │ A²[1]-Y[1]  ...  ...   │    =  A²  -  Y_onehot
  │    ...       ...  ...  │
  │A²[9]-Y[9]   ...  ...   │
  └                        ┘
  Each cell = (predicted prob) - (true prob)
  Correct class: was 0.44, should be 1.0  →  dZ² = -0.56  (big negative → push up)
  Wrong  class:  was 0.09, should be 0.0  →  dZ² = +0.09  (small positive → push down)
```

**Step 2 — Gradients for W², b²:**
```
  dW²  (10 × 128)          =  (1/m) · dZ² (10×m)  ·  A¹ᵀ (m×128)

  db²  (10 × 1)            =  (1/m) · sum(dZ², axis=1)
```

**Step 3 — Push error back to hidden layer:**
```
  dA¹  (128 × m)           =  W²ᵀ (128×10)  ·  dZ² (10×m)

  dZ¹  (128 × m)           =  dA¹  ⊙  ReLU'(Z¹)
                                        ↑
                               element-wise multiply
                               1 where Z¹>0 (gradient flows)
                               0 where Z¹≤0 (gradient blocked)

  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  0.3  -0.1  0.5  │     │   1    0    1    │     │  0.3   0    0.5  │
  │ -0.2   0.4 -0.3  │  ⊙  │   0    1    0    │  =  │  0     0.4   0   │
  │  0.1   0.2  0.7  │     │   1    1    1    │     │  0.1   0.2  0.7  │
  └──────────────────┘     └──────────────────┘     └──────────────────┘
      dA¹ (gradient)           ReLU gate                 dZ¹ (blocked)
                            (from forward pass)
```

**Step 4 — Gradients for W¹, b¹:**
```
  dW¹  (128 × 784)         =  (1/m) · dZ¹ (128×m)  ·  Xᵀ (m×784)
  [shape: 128×784, same as W¹]

  db¹  (128 × 1)           =  (1/m) · sum(dZ¹, axis=1)
```

### Why it's necessary
A network with 128 hidden neurons on 784 inputs has `128×784 + 128 + 10×128 + 10 = 101,770` parameters. There is no other tractable way to compute the gradient for each one. Backprop reuses intermediate results (dZ², dA¹) so the cost is proportional to the number of parameters — not exponential.

### The ReLU gate in backprop
`dZ¹ = dA¹ ⊙ ReLU'(Z¹)` — neurons that were silent during the forward pass get **zero gradient**. They don't update this step. The forward pass directly controls which neurons learn.

### What would break without it
No way to assign blame to individual weights. The network cannot improve.

---

## Component 7 — Gradient Descent

### The Problem It Solves

After the forward pass, we have a loss — a single number saying "the network is this wrong."
After backprop, we have gradients — a number for every weight saying "if this weight increases slightly, the loss changes by this much."

But knowing the loss is wrong and knowing the gradient **does not automatically fix the network**. Something has to actually change the weights. That is the sole job of Gradient Descent.

> **Gradient Descent is the mechanism that translates the mathematical signal from backprop into actual weight updates. Without it, the network computes how to improve but never does.**

---

### Intuition — The Hill Analogy

Imagine you are blindfolded on a hilly landscape and your goal is to reach the lowest valley (minimum loss). You can't see the full landscape. The only thing you can feel is the **slope under your feet right now** — that's the gradient.

```
  Loss (height)
   │
   │       ●  ← you are here (current weights, high loss)
   │      /│\
   │     / │ \
   │    /  ↓  \       gradient tells you the slope
   │   /   ↓   \      you step downhill (opposite to gradient)
   │  /    ↓    \
   │ /     ●     \    ← lower loss after one step
   │/             \
   │               ●  ← minimum (lowest loss)
   └──────────────────── weight value
```

The gradient tells you which direction is uphill. You go **opposite** to it — downhill — by a controlled step size `α`. This is why the update rule has a **minus sign**:

```
W := W - α · dW
       ↑
    minus = move opposite to gradient = move downhill
```

---

### What it does
Applies the gradient to every weight after each backward pass:

```
W := W - α · dW
b := b - α · db
```

**Concrete update for W²** `(10 × 128)`:

```
  W²  (10 × 128)              α · dW²  (10 × 128)          W²_new  (10 × 128)
  [actual shape: 10×128]       [actual shape: 10×128]        [actual shape: 10×128]
  ┌                 ┐          ┌                 ┐           ┌                 ┐
  │ 0.31  -0.12 ... │          │ 0.02  -0.01 ... │           │ 0.29  -0.11 ... │
  │-0.44   0.55 ... │    -     │-0.03   0.04 ... │     =     │-0.41   0.51 ... │
  │  ...   ...  ... │          │  ...   ...  ... │           │  ...   ...  ... │
  └                 ┘          └                 ┘           └                 ┘
  current weights              small nudge                   updated weights
                               (α=0.1 keeps it small)
```

The same happens simultaneously for all four parameter matrices: `W¹, b¹, W², b²`.
In our network that's **101,770 individual weight updates** — every single iteration.

---

### Why the Minus Sign Is Everything

The gradient `dW` points in the direction of **steepest increase** in loss.
We want to **decrease** loss, so we move in the exact opposite direction.

```
  dW > 0  →  increasing W increases loss  →  W := W - α·dW  (decreases W)
  dW < 0  →  decreasing W increases loss  →  W := W - α·dW  (increases W)
  dW = 0  →  W is at a local minimum      →  no change needed
```

This is why Gradient Descent always makes progress — it is mathematically guaranteed to reduce loss at each step, as long as the learning rate is sensible.

---

### The Role of the Learning Rate α

The gradient tells you **direction**. The learning rate `α` controls **how far** you step in that direction. This is critical:

```
  Loss
   │
   │  ●  ← start (high loss)
   │   \
   │    ●  α too large: step overshoots the minimum, bounces around
   │   / \
   │  ●   ●   (loss may even increase)
   │
   │  ●  ← start
   │   ↘
   │    ↘  α just right: steady, controlled descent
   │     ↘
   │      ●  ← converged (low loss)
   │
   │  ● ← start
   │  ↓  α too small: steps are tiny, takes thousands of iterations
   │  ●
   └──────────────────────────── iterations
```

| Learning rate | Effect |
|---|---|
| Too large (e.g. 1.0) | Overshoots minimum, loss oscillates or diverges |
| Too small (e.g. 0.0001) | Crawls to minimum, takes too long |
| Just right (e.g. 0.1) | Steadily descends to a good minimum |

There is no formula for the perfect `α` — it is found by experimentation. This is one of the most important hyperparameters you tune when training a network.

---

### The Full Picture — Where Gradient Descent Sits

```
  Forward Pass  →  Loss  →  Backprop  →  Gradient Descent
                    │              │              │
               "how wrong     "who is        "fix them"
                are we?"      responsible?"
```

Every other component feeds into Gradient Descent. The loss gives it a signal to minimise. Backprop gives it the precise gradients to act on. Gradient Descent is the **executor** — the step that actually changes the network.

Without it, the network is a read-only system. It can measure how wrong it is, compute exactly what needs to change, and then do absolutely nothing.

### What would break without it
Gradients computed but never applied — the network stays at random initialisation forever, loss never decreases, accuracy stays at ~10%.

---

## Why the Order Matters

Each component feeds directly into the next. Remove any one and the chain breaks:

| Remove this | What breaks |
|---|---|
| Weights (neurons) | Nothing to learn — network is fixed |
| Layers | No feature hierarchy — can't solve complex problems |
| ReLU | Network collapses to linear — can only fit straight lines |
| Softmax | Output isn't a probability — loss is numerically undefined |
| Loss function | No differentiable signal — backprop has nothing to compute |
| Backpropagation | Can't assign credit to individual weights — no learning |
| Gradient descent | Gradients computed but never applied — no improvement |

---

## The Training Loop in One Sentence

> For each iteration: make a prediction (forward), measure the error (loss), figure out who's responsible (backprop), and fix them (gradient descent) — then repeat until the network is good enough.

That's all a neural network is.

---

## Full Matrix Flow — One Forward + Backward Pass

```
INPUT
  X         (784 × m)   ← 784 pixel values per sample, m samples

HIDDEN LAYER — forward
  Z¹ = W¹·X + b¹        W¹ (128×784),  b¹ (128×1)  →  Z¹ (128×m)
  A¹ = ReLU(Z¹)         element-wise max(0,·)        →  A¹ (128×m)

OUTPUT LAYER — forward
  Z² = W²·A¹ + b²       W² (10×128),   b²  (10×1)  →  Z² (10×m)
  A² = Softmax(Z²)      column-wise normalise         →  A² (10×m)

LOSS
  L  = -mean(log(A²[Y, :]))                          →  L  (scalar)

OUTPUT LAYER — backward
  dZ²= A² - Y_onehot                                 →  dZ² (10×m)
  dW²= (1/m) dZ² · A¹ᵀ                              →  dW² (10×128)
  db²= (1/m) sum(dZ², axis=1)                        →  db² (10×1)

HIDDEN LAYER — backward
  dA¹= W²ᵀ · dZ²                                    →  dA¹ (128×m)
  dZ¹= dA¹ ⊙ ReLU'(Z¹)                              →  dZ¹ (128×m)
  dW¹= (1/m) dZ¹ · Xᵀ                               →  dW¹ (128×784)
  db¹= (1/m) sum(dZ¹, axis=1)                        →  db¹ (128×1)

WEIGHT UPDATE
  W² -= α · dW²         W¹ -= α · dW¹
  b² -= α · db²         b¹ -= α · db¹
```

Every shape is intentional — the dimensions must align for each multiply to be valid, and they always produce a gradient with the **exact same shape** as the weight it will update.

---

## What We Built vs What Keras Does

| Ours | Keras equivalent |
|---|---|
| `class Neuron` | Internal weight initialization |
| `class Layer` | `Dense(units, ...)` |
| `ReLU(Z)` | `activation='relu'` |
| `softmax(Z)` | `activation='softmax'` |
| `cross_entropy_loss()` | `loss='sparse_categorical_crossentropy'` |
| `backward()` | `model.fit()` auto-differentiates |
| `train()` loop | `model.fit(epochs=...)` |

Keras wraps all of this with automatic differentiation (no need to derive gradients by hand) and hardware acceleration — but the math and matrix shapes underneath are **identical** to what you built here.
