import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output, display
from matplotlib.patches import Patch


def generate_dissipative_pendulum_data(t_span=(0, 100), dt=0.1, spring_const=5.0, linear_drag=0.2, drag_force=None, x0=2.0, v0=0.0, noise_std=0.0):
    """
    Generate synthetic data from a dissipative pendulum with quadratic drag. (Assume mass=1 for simplicity).
    
    ODE: 
        dv/dt = -k*x - c0*v + drag_force(v)
        dx/dt = v
    
    Args:
        t_span: Tuple of (t_start, t_end)
        dt: Time step
        spring_const: Spring constant k (restoring force coefficient)
        linear_drag: Linear damping coefficient c0
        drag_force: Callable function of velocity for additional drag (e.g., quadratic drag)
        x0: Initial position
        v0: Initial velocity
        noise_std: Standard deviation of Gaussian noise to add to observations
    
    Returns:
        time: Array of time points (shape: [n_points])
        positions: Array of positions (shape: [n_points, 1])
        velocities: Array of velocities (shape: [n_points, 1])
        accelerations: Array of accelerations (shape: [n_points, 1])
    """

    drag_force = drag_force if drag_force is not None else (lambda v: -0.0 * v)

    def ode_rhs(t, state):
        x, v = state
        dvdt = -spring_const * x - linear_drag * v + drag_force(v)
        dxdt = v
        return [dxdt, dvdt]
    
    # Time array
    time = np.arange(t_span[0], t_span[1], dt)
    
    # Integrate ODE using RK45
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode_rhs, t_span, [x0, v0], t_eval=time, method='RK45', rtol=1e-9, atol=1e-12)
    
    positions = sol.y[0, :].reshape(-1, 1)
    velocities = sol.y[1, :].reshape(-1, 1)
    
    # Compute accelerations from the ODE
    accelerations = np.zeros_like(positions)
    for i in range(len(time)):
        x, v = positions[i, 0], velocities[i, 0]
        accelerations[i, 0] = -spring_const * x - linear_drag * v + drag_force(v)
    
    # Add noise if requested
    if noise_std > 0:
        positions += np.random.normal(0, noise_std, positions.shape)
        velocities += np.random.normal(0, noise_std, velocities.shape)
        accelerations += np.random.normal(0, noise_std, accelerations.shape)
    
    return time, positions, velocities, accelerations


def split_and_plot_dataset(data, train_frac=0.4, val_frac=None, show_plot=True):
    """
    Split t,x,v,a into train/val and plot the entire dataset with highlighted regions.
    
    Args:
        data: Tuple of (time, positions, velocities, accelerations)
        train_frac: Fraction of data for training (default: 0.4)
        val_frac: Fraction of data for validation. If None, uses remaining data (1 - train_frac)
        show_plot: Whether to display the plot (default: True)
    
    Returns:
        If show_plot=True: ((train_x, train_v, train_a, train_t), (val_x, val_v, val_a, val_t))
        If show_plot=False: (train_x, train_v, train_a, train_t, val_x, val_v, val_a, val_t)
    """

    if val_frac is None:
        val_frac = 1 - train_frac
    t, x, v, a = data
    n_samples = len(t)
    train_end = int(train_frac * n_samples)
    val_end = int((train_frac + val_frac) * n_samples)

    train_x, train_v, train_a, train_t = x[:train_end], v[:train_end], a[:train_end], t[:train_end]
    val_x, val_v, val_a, val_t = x[train_end:val_end], v[train_end:val_end], a[train_end:val_end], t[train_end:val_end]

    if not show_plot:
        return (train_x, train_v, train_a, train_t,
                val_x, val_v, val_a, val_t)

    # Define region boundaries in time (robust to tiny edge cases)
    t0 = float(t[0])
    t_train_end = float(t[train_end - 1]) if train_end > 0 else t0
    t_val_start = float(t[train_end]) if train_end < n_samples else t_train_end
    t_val_end = float(t[val_end - 1]) if val_end > train_end else t_val_start

    train_color = 'C0'
    val_color = 'C1'


    fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True)

    # Position
    axes.plot(t, x.squeeze(), 'r', lw=2, label='Position')
    axes.plot(t, v.squeeze(), 'b', lw=2, label='Velocity')
    axes.plot(t, a.squeeze(), 'g', lw=2, label='Acceleration')
    axes.axvspan(t0, t_train_end, color=train_color, alpha=0.2)
    axes.axvspan(t_val_start, t_val_end, color=val_color, alpha=0.2)
    axes.grid(True, alpha=0.3)
    # Shared legend
    legend_handles = [
        Patch(facecolor=train_color, alpha=0.2, label='Train region'),
        Patch(facecolor=val_color, alpha=0.2, label='Validation region'),
        plt.Line2D([0], [0], color='r', lw=2, label='$x(t)$ (Position)'),
        plt.Line2D([0], [0], color='b', lw=2, label='$\\dot{x}(t)$ (Velocity)'),
        plt.Line2D([0], [0], color='g', lw=2, label='$\\ddot{x}(t)$ (Acceleration)'),
    ]
    axes.legend(handles=legend_handles, loc='upper right')
    axes.set_title('Training + validation data')
    plt.tight_layout()
    plt.show()

    return (train_x, train_v, train_a, train_t), (val_x, val_v, val_a, val_t)

def plot_loss_curve(ax, train_loss_list, val_loss_list, num_epochs):
    """
    Plot training and validation loss curves on a log scale.
    
    Args:
        ax: Matplotlib axis to plot on
        train_loss_list: List of training losses per epoch
        val_loss_list: List of validation losses per epoch
        num_epochs: Total number of epochs (for x-axis limit)
    """
    ax.semilogy(train_loss_list, label="Training Loss", alpha=0.7, linewidth=2.5, color="tab:blue")
    ax.semilogy(val_loss_list, label="Validation Loss", alpha=0.7, linewidth=2.5, color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(loc='upper right', fontsize=8)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_epochs)
    ax.set_yscale("log")

## physics informed ode utils: 

def to_tensor(array, *, device=None):
    """
    Convert numpy array to torch tensor.
    
    Args:
        array: Numpy array to convert
        device: Torch device to place tensor on (optional)
    
    Returns:
        Torch tensor on specified device (or CPU if device is None)
    """
    tensor = torch.tensor(array, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def train_vector_field(
    model,
    data_train,
    data_val,
    device=None,
    num_epochs=5000,
    print_every=500,
    lr=0.01,
    best_val=float("inf"),
    generate_gif=False,
    gif_filename="training_progress.gif",
):
    """
    Train a vector field model to learn accelerations from positions and velocities.
    
    Args:
        model: Physics-informed vector field model (must have forward(x, v) returning accelerations)
        data_train: Tuple of (train_x, train_v, train_a, train_t) training data
        data_val: Tuple of (val_x, val_v, val_a, val_t) validation data
        device: Torch device (default: 'cpu')
        num_epochs: Number of training epochs (default: 5000)
        print_every: Frequency of printing/plotting progress (default: 500)
        lr: Learning rate (default: 0.01)
        best_val: Initial best validation loss (default: inf)
        generate_gif: Whether to save training progress as animated GIF (default: False)
        gif_filename: Filename for the GIF if generate_gif=True (default: 'training_progress.gif')
    
    Returns:
        Dictionary containing:
            - 'model': Trained model with best validation loss
            - 'best_val': Best validation loss achieved
            - 'train_loss_history': List of training losses per epoch
            - 'val_loss_history': List of validation losses per epoch
    """
    try:

        if device is None:
            device = "cpu"
        
        # List to store frames for GIF generation
        gif_frames = [] if generate_gif else None

        # Unpack grouped data
        train_x, train_v, train_a, train_t = data_train
        val_x, val_v, val_a, val_t = data_val

        def to_tensor_local(arr):
            return torch.tensor(arr, dtype=torch.float32, device=device)

        train_x_t = to_tensor_local(train_x)
        train_v_t = to_tensor_local(train_v)
        train_a_t = to_tensor_local(train_a)

        val_x_t = to_tensor_local(val_x)
        val_v_t = to_tensor_local(val_v)
        val_a_t = to_tensor_local(val_a)
        
        
        # Setup optimizer and loss
        best_state = copy.deepcopy(model.state_dict())
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_loss_history = []
        val_loss_history = []

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass: predict accelerations from positions and velocities
            pred_a = model(train_x_t, train_v_t)
            
            # Compute loss: ||a_predicted - a_true||^2
            loss = criterion(pred_a, train_a_t)
                        
            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                pred_a_val = model(val_x_t, val_v_t)
                val_loss = criterion(pred_a_val, val_a_t)
                val_loss_history.append(val_loss.item())
                
                # Save best model
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    best_state = copy.deepcopy(model.state_dict())
            
            # Print and plot progress
            if (epoch + 1) % print_every == 0:
                title = f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {loss.item():.2e} | Val Loss: {val_loss.item():.2e}"
                clear_output(wait=True)
                
                # Create figure with rollout predictions and loss curve
                state_dim = val_x.shape[1]
                if state_dim == 1:
                    # Single pendulum case
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                    
                    # Compute rollout on validation set
                    pred_x_val, _ = rollout_vector_field(model, val_x[0:1], val_v[0:1], val_t)
                    
                    # Position plot
                    axes[0].plot(val_t, val_x[:, 0], label='Validation data',
                           color='tab:orange', linewidth=2, alpha=0.8)
                    axes[0].plot(val_t, pred_x_val[:, 0], label='Prediction', 
                           color='black', linestyle='--', linewidth=2.5, alpha=0.7)
                    axes[0].set_ylabel('Position')
                    axes[0].set_xlabel('Time [s]')
                    axes[0].grid(True, alpha=0.3)
                    axes[0].legend(loc='upper right', fontsize=8)
                    
                    # Loss curve
                    plot_loss_curve(axes[1], train_loss_history, val_loss_history, num_epochs)
                
                plt.suptitle(title, fontsize=12)
                plt.tight_layout()
                
                # Save frame for GIF if requested
                if generate_gif:
                    # Convert figure to numpy array
                    fig.canvas.draw()
                    # Use buffer_rgba() for newer matplotlib versions
                    width, height = fig.canvas.get_width_height()
                    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    frame = frame.reshape((height, width, 4)).copy()[:, :, :3]  # Copy to detach buffer
                    gif_frames.append(frame)
                
                display(fig)
                plt.close()
        
        # Load best model
        model.load_state_dict(best_state)
        print(f"\nTraining complete! Returning model with best validation loss: {best_val:.6e}")
        
        # Generate GIF if requested
        if generate_gif and gif_frames:
            try:
                import imageio
                imageio.mimsave(gif_filename, gif_frames, fps=10, loop=0)
                print(f"Training progress GIF saved to: {gif_filename}")
            except ImportError:
                print("Warning: imageio not installed. Install with 'pip install imageio' to generate GIFs.")
            except Exception as e:
                print(f"Warning: Could not save GIF. Error: {e}")
        
    except KeyboardInterrupt:
        print("Training interrupted. Returning the best model found so far.")
        model.load_state_dict(best_state)
        
        # Save GIF even if interrupted
        if generate_gif and gif_frames:
            try:
                import imageio
                imageio.mimsave(gif_filename, gif_frames, fps=2, loop=0)
                print(f"Training progress GIF saved to: {gif_filename}")
            except Exception as e:
                print(f"Warning: Could not save GIF. Error: {e}")
    
    return {
        "model": model,
        "best_val": best_val,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }


def rollout(model, x0, v0, t0, num_steps):
    """
    Perform multi-step rollout using learned discrete flow map.
    
    Args:
        model: Flow map model with forward(x, v, t) returning (x_next, v_next)
        x0: Initial position tensor [batch_size, state_dim]
        v0: Initial velocity tensor [batch_size, state_dim]
        t0: Initial time
        num_steps: Number of steps to rollout
    
    Returns:
        predicted_x: Array of predicted positions [num_steps, state_dim]
        predicted_v: Array of predicted velocities [num_steps, state_dim]
    """
    model.eval()

    predicted_x = [x0.cpu().numpy().squeeze()]
    predicted_v = [v0.cpu().numpy().squeeze()]

    x_curr, v_curr, t_curr = x0, v0, t0
    dt = model.dt

    with torch.no_grad():
        for _ in range(num_steps-1):
            x_next, v_next = model(x_curr, v_curr, t_curr)
            predicted_x.append(x_next.cpu().numpy().squeeze())
            predicted_v.append(v_next.cpu().numpy().squeeze())
            x_curr, v_curr = x_next, v_next
            t_curr = t_curr + dt

    return np.array(predicted_x), np.array(predicted_v)

def plot_predictions(model, test_x, test_v, test_t, title=None, fourth_ax=None):
    """
    Plot multi-dimensional rollout predictions (3D system) with optional fourth subplot.
    
    Args:
        model: Flow map model with rollout capability
        test_x: Ground truth positions [num_steps, 3]
        test_v: Ground truth velocities [num_steps, 3]
        test_t: Time array [num_steps]
        title: Optional title for the plot
        fourth_ax: Optional function to populate fourth subplot
    """
    colors = ["tab:blue", "tab:orange", "tab:green"]
    model.eval()
    num_test_steps = test_x.shape[0]
    test_x_init = to_tensor(test_x[0:1], device=next(model.parameters()).device)
    test_v_init = to_tensor(test_v[0:1], device=next(model.parameters()).device)
    test_t_init = to_tensor(test_t[0:1], device=next(model.parameters()).device)
    pred_x_test, pred_v_test = rollout(model, test_x_init, test_v_init, test_t_init, num_test_steps)

    if fourth_ax is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharex='col')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3), sharex='col')

    component_labels = ['Surge', 'Heave', 'Pitch']
    for i in range(3):
        axes[i].plot(test_t, test_x[:, i], label='Ground Truth', color=colors[i], linewidth=2.5)
        axes[i].plot(test_t, pred_x_test[:, i], label='Prediction (rollout)', color='k', linestyle='--', linewidth=2.5, alpha=0.7)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Time [s]')
        axes[i].set_ylabel(component_labels[i])
        axes[i].set_ylim(np.min(test_x[:, i])*1.5, np.max(test_x[:, i])*1.5)
        if i == 0:
            axes[i].legend(loc='center', bbox_to_anchor=(0.5, 1.2))

    if fourth_ax is not None:
        fourth_ax(axes[3])
        

    plt.tight_layout()
    plt.suptitle(title, fontsize=14)


def rollout_vector_field(model, x0, v0, time_array):
    """
    Perform rollout by integrating the learned vector field using RK4 integration.
    
    Args:
        model: Physics-informed model with forward(x, v) method returning accelerations
        x0: Initial positions [1, state_dim] or [state_dim] tensor
        v0: Initial velocities [1, state_dim] or [state_dim] tensor
        time_array: Array of time points to evaluate at [num_steps]
    
    Returns:
        predicted_x: Array of predicted positions [num_steps, state_dim]
        predicted_v: Array of predicted velocities [num_steps, state_dim]
    
    Notes:
        Uses 4th-order Runge-Kutta (RK4) integration for accuracy.
    """
    model.eval()
    
    # Ensure inputs are tensors with correct shape
    if isinstance(x0, np.ndarray):
        x0 = torch.tensor(x0, dtype=torch.float32, device=next(model.parameters()).device)
    if isinstance(v0, np.ndarray):
        v0 = torch.tensor(v0, dtype=torch.float32, device=next(model.parameters()).device)
    
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    if v0.dim() == 1:
        v0 = v0.unsqueeze(0)
    
    predicted_x = []
    predicted_v = []

    x_curr = x0
    v_curr = v0

    if len(time_array) < 2:
        dt = 0.0
    else:
        dt = float(time_array[1] - time_array[0])

    def rk4_step(x, v, step):
        # Compute Runge-Kutta terms for x' = v, v' = a(x, v)
        a1 = model(x, v)
        k1_x = v
        k1_v = a1

        a2 = model(x + 0.5 * step * k1_x, v + 0.5 * step * k1_v)
        k2_x = v + 0.5 * step * k1_v
        k2_v = a2

        a3 = model(x + 0.5 * step * k2_x, v + 0.5 * step * k2_v)
        k3_x = v + 0.5 * step * k2_v
        k3_v = a3

        a4 = model(x + step * k3_x, v + step * k3_v)
        k4_x = v + step * k3_v
        k4_v = a4

        x_next = x + (step / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v_next = v + (step / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return x_next, v_next

    with torch.no_grad():
        for i in range(len(time_array)):
            predicted_x.append(x_curr.cpu().numpy()[0])
            predicted_v.append(v_curr.cpu().numpy()[0])

            if i < len(time_array) - 1 and dt > 0.0:
                x_curr, v_curr = rk4_step(x_curr, v_curr, dt)

    return np.array(predicted_x), np.array(predicted_v)


def plot_vector_field_predictions(model, data_test, title=None, filename=None):
    """
    Plot predictions from a vector field model against ground truth for 1D systems.
    
    Args:
        model: Physics-informed vector field model with forward(x, v) method
        data_test: Tuple of (test_t, test_x, test_v, test_a) containing test data
        title: Optional title for the plot
    
    Notes:
        Integrates the learned vector field from initial conditions and compares to ground truth.
    """
    model.eval()
    
    # Get initial conditions
    test_t, test_x, test_v, _ = data_test
    x0 = test_x[0:1] if isinstance(test_x, np.ndarray) else test_x[0:1].cpu().numpy()
    v0 = test_v[0:1] if isinstance(test_v, np.ndarray) else test_v[0:1].cpu().numpy()
    
    # Rollout the model
    pred_x, pred_v = rollout_vector_field(model, x0, v0, test_t)
        
    # Single pendulum case
    fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True)
    
    # Plot position
    axes.plot(test_t, test_x[:, 0], label='Ground Truth', 
                    color='tab:blue', linewidth=2.5, alpha=0.8)
    axes.plot(test_t, pred_x[:, 0], label='Prediction', 
                    color='black', linestyle='--', linewidth=2, alpha=0.7)
    axes.set_ylabel('Position')
    axes.grid(True, alpha=0.3)
    axes.legend(loc='upper right')
    
    if title:
        plt.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')  
    plt.show()
