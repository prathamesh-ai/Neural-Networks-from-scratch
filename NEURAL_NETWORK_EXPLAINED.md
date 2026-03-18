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
Computes a weighted sum of its inputs plus a bias:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Why it's necessary
A neuron is the unit of **parameterized computation**. The weights `w` and bias `b` are the things the network *learns*. Without neurons — without weights — there is nothing to adjust, nothing to train, nothing to learn.

The bias `b` is critical: it lets a neuron activate even when all inputs are zero, giving the neuron its own "default opinion" independent of the data.

### What would break without it
Without weights, the network would be a fixed transformation — the same for every problem, with no ability to adapt.

---

## Component 2 — The Layer (Matrix Math)

### What it does
A layer is N neurons operating in parallel on the same input, expressed as a matrix multiplication:

```
Z = W · X + b
```
- `W` shape: `(n_neurons, n_inputs)` — one row of weights per neuron
- `X` shape: `(n_inputs, m)` — m samples processed simultaneously
- `Z` shape: `(n_neurons, m)` — every neuron's output for every sample

### Why it's necessary
Two reasons:

1. **Efficiency**: A single matrix multiply computes all N neurons for all M samples in one operation. This is why neural networks run fast on GPUs — they are massively parallel matrix machines.

2. **Feature hierarchy**: Each layer learns a different level of abstraction. Layer 1 might detect edges in an image. Layer 2 combines edges into shapes. Layer 3 combines shapes into objects. You cannot build this hierarchy with a single neuron.

### What would break without it
Without layers, you'd need to manually loop over every neuron for every sample — unusably slow, and no hierarchy of features.

---

## Component 3 — ReLU Activation Function

### What it does
Applied element-wise after every hidden layer:

```
ReLU(z) = max(0, z)
```

Negative values → 0. Positive values → unchanged.

### Why it's necessary
This is the most misunderstood component. The zero-ing of negatives **is not** the point. The point is **non-linearity**.

Without activation functions, no matter how many layers you stack, the entire network collapses mathematically into a single linear equation:

```
Layer 2(Layer 1(X)) = (W₂W₁)X + (W₂b₁ + b₂) = W'X + b'
```

That's just one straight line. A straight line can never learn:
- A curve
- A boundary that isn't flat
- Any real-world pattern of meaningful complexity

ReLU breaks this by treating the positive and negative regions differently. Each neuron effectively becomes a **switch** — off for its dead zone, linear for its active zone. A layer of neurons with different weights and biases places these switches at different points, and their weighted sum approximates any curve.

### Why ReLU specifically (vs other activations)
- Simple to compute: `max(0, z)` is one operation
- Simple derivative: 1 where active, 0 where dead — clean gradient flow
- No vanishing gradient for positive inputs (unlike Sigmoid/Tanh which saturate)

### What would break without it
Without activation functions: no matter how deep the network, it can only fit straight lines. It would never reach 85% accuracy on MNIST — it would plateau around 30-40%.

---

## Component 4 — Softmax (Output Activation)

### What it does
Converts raw output scores (logits) into a valid probability distribution:

```
Softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

All outputs sum to exactly 1. The largest logit gets the highest probability.

### Why it's necessary
The output layer produces raw numbers — they could be 100, -50, 0.3. These are meaningless as probabilities.

We need probabilities because:
1. The prediction is `argmax(output)` — we need to compare across classes fairly
2. The loss function (cross-entropy) requires probabilities — it takes `log(probability)` of the correct class
3. Probabilities give us **confidence** — not just "class 3" but "87% sure it's class 3"

### Why not ReLU on the output?
ReLU outputs can be 0 — a class could have exactly 0 probability, which breaks `log(0)` in the loss. ReLU outputs also don't sum to 1, so you can't interpret them as probabilities.

### What would break without it
Raw logits fed into cross-entropy would produce meaningless loss values. Training would be numerically unstable or produce nonsensical gradients.

---

## Component 5 — Loss Function (Cross-Entropy)

### What it does
Measures how wrong the network's prediction is with a single number:

```
L = -log(probability assigned to the correct class)
```

- Predicted `p=0.99` for correct class → loss = `0.01` (nearly perfect)
- Predicted `p=0.10` for correct class → loss = `2.30` (random guessing)
- Predicted `p=0.01` for correct class → loss = `4.60` (confidently wrong)

### Why it's necessary
Training requires a **differentiable** measure of error. Accuracy (0 or 1 per sample) is not differentiable — you can't compute "which direction should I move the weights to increase accuracy by a tiny amount?".

Cross-entropy is smooth and differentiable everywhere. Its gradient tells the network precisely how to adjust to improve.

The `-log` shape is intentional: it punishes overconfident wrong predictions **disproportionately hard**. A network that says "I'm 99% sure it's a 3" when it's actually a 7 gets a loss of 4.60. This forces the network to be calibrated — not just right, but appropriately confident.

### What would break without it
Without a differentiable loss, there is no gradient, and without a gradient, there is no backpropagation. The network cannot learn.

---

## Component 6 — Backpropagation

### What it does
Computes the gradient of the loss with respect to **every single weight** in the network, using the Chain Rule:

```
∂L/∂W = ∂L/∂A · ∂A/∂Z · ∂Z/∂W
```

Walking backwards through the network, each layer receives the error signal from the layer ahead of it and passes a transformed version back to the layer behind it.

For our two-layer network:
```
dZ2 = A2 - Y_onehot          ← error at output (how far off were the probabilities?)
dW2 = (1/m) dZ2 · A1ᵀ        ← how much did W2 contribute to that error?
dA1 = W2ᵀ · dZ2              ← push error back through to hidden layer
dZ1 = dA1 ⊙ ReLU'(Z1)        ← ReLU gate: block gradient where neuron was dead
dW1 = (1/m) dZ1 · Xᵀ         ← how much did W1 contribute?
```

### Why it's necessary
A network has thousands of weights. We need to know the gradient for **each one** to know how to update it. You cannot compute this by hand or by trying random updates.

Backprop is an efficient application of the chain rule that reuses intermediate computations, making gradient calculation tractable even for millions of parameters.

### The ReLU gate in backprop
Notice `dZ1 = dA1 ⊙ ReLU'(Z1)`. ReLU's derivative is 1 where active, 0 where dead.
This means: neurons that were silent during the forward pass receive **zero gradient** during backprop. They don't update. This is the direct connection between forward and backward passes — the forward pass determines which neurons participate in learning.

### What would break without it
Without backprop, we'd have no way to assign credit or blame to individual weights. The network could not improve.

---

## Component 7 — Gradient Descent

### What it does
Uses the gradients computed by backprop to update every weight:

```
W := W - α · dW
b := b - α · db
```

`α` (alpha) is the **learning rate** — it controls step size.

### Why it's necessary
Knowing the gradient tells you the direction of steepest ascent on the loss surface. Moving **opposite** to the gradient (hence the minus sign) moves you downhill — toward lower loss.

### The learning rate trade-off
| Learning rate | Effect |
|---|---|
| Too large (e.g. 1.0) | Overshoots the minimum, loss oscillates or diverges |
| Too small (e.g. 0.0001) | Crawls toward minimum, takes forever to train |
| Just right (e.g. 0.1) | Steadily descends, reaches a good minimum |

### What would break without it
Gradients alone tell you direction but not magnitude of the step. Without a controlled update rule, the network would wildly overshoot or never converge.

---

## Why the Order Matters

Each component feeds into the next. Remove any one of them and the chain breaks:

| Remove this | What breaks |
|---|---|
| Weights (neurons) | Nothing to learn — network is fixed |
| Layers | No feature hierarchy — can't solve complex problems |
| ReLU | Network collapses to linear — can't fit curves |
| Softmax | Output isn't a probability — loss is meaningless |
| Loss function | No differentiable signal — backprop has nothing to compute |
| Backpropagation | Can't assign credit to individual weights — no learning |
| Gradient descent | Gradients computed but never applied — no improvement |

---

## The Training Loop in One Sentence

> For each iteration: make a prediction (forward), measure the error (loss), figure out who's responsible (backprop), and fix them (gradient descent) — then repeat until the network is good enough.

That's all a neural network is.

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

Keras wraps all of this with automatic differentiation (no need to derive gradients by hand) and hardware acceleration — but the math underneath is identical to what you built here.
