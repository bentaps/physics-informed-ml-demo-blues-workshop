# Results: Training Progress and Model Comparison

You can run this notebook on Google collab by pressing this link: [tinyurl.com/blues-ai-workshop](https://tinyurl.com/blues-ai-workshop)

Below are animated GIFs showing the training loss curves and validation predictions for:

**Vanilla Neural ODE model (no physics):**
<p align="center">
	<img src="figs/training_progress_vanilla.gif" alt="Vanilla Model Training" width="62%">
	<img src="figs/vanilla_model_test_set_predictions.png" alt="Vanilla Model Training" width="36%">
</p>
 
**Physics-Informed Neural ODE model:**
<p align="center">
	<img src="figs/training_progress_physics.gif" alt="Physics-Informed Model Training" width="62%">
	<img src="figs/physics_model_test_set_predictions.png" alt="Physics-Informed Model Training" width="36%">
</p>


Notice how the physics model achieves lower validation error and more stable long-term predictions, even with limited data.

# Physics-Informed Machine Learning: Learning Dynamical Systems

Learn how to combine physical knowledge with machine learning to estimate system parameters and predict dynamics from sparse data.

## Quick Start

### Main Notebook
**`physics_informed_ode.ipynb`** — Complete pedagogical walkthrough demonstrating:
- **Vanilla Neural Network**: Pure data-driven baseline (poor generalization)
- **Physics-Informed Model**: Decomposes acceleration into known physics + learned residuals
- **Parameter Recovery**: Estimates spring constant $K$ and damping coefficient $C_0$ from data

### Core Idea

Instead of learning the full acceleration $\ddot{x} = f(x, \dot{x})$ with a neural network, we use physics:

$$\ddot{x} = \underbrace{-Kx - C_0\dot{x}}_{\text{Known Physics}} + \underbrace{f(\dot{x})}_{\text{Learned Residual}}$$

This structure dramatically improves:
- **Data efficiency**: Fewer parameters to learn
- **Interpretability**: $K$ and $C_0$ have physical meaning
- **Generalization**: Physics constraints stabilize long-term predictions

## Key Files

- **`physics_informed_ode.ipynb`** — Tutorial notebook (start here)
- **`models.py`** — Model definitions (`VanillaModel`, `PhysicsInformedModel`)
- **`utils.py`** — Training loops and visualization utilities

## Installation

```bash
pip install -r requirements.txt
```

Run the notebook in Jupyter:
```bash
jupyter notebook physics_informed_ode.ipynb
```

## Learning Objectives

- Understand when and why physics-informed learning outperforms pure data-driven approaches
- Apply this framework to your own dynamical systems (ODEs, PDEs, control systems, etc.)
- Recover unknown physical parameters from experimental data
