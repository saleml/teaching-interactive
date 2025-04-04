import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import pandas as pd


class LineEnvironment:
    def __init__(
        self,
        p_left_initial: float = 0.5,
        time_homogeneous: bool = True,
        time_decay: float = 0.0,
    ):
        """Initialize the random walk environment.

        Args:
            p_left_initial: Probability of moving left at t=0.
            time_homogeneous: If True, p_left is constant. If False, it decays.
            time_decay: The decay factor (lambda) if non-homogeneous.
                      p_left(t) = p_left_initial * exp(-time_decay * t)
        """
        if not 0.0 <= p_left_initial <= 1.0:
            raise ValueError("p_left_initial must be between 0 and 1")
        if time_decay < 0.0:
            raise ValueError("time_decay cannot be negative")

        self.p_left_initial = p_left_initial
        self.time_homogeneous = time_homogeneous
        self.time_decay = time_decay

        self.state_min = -1000
        self.state_max = 1000

    def get_p_left(self, t: int) -> float:
        """Calculate the probability of moving left at time t."""
        if self.time_homogeneous:
            return self.p_left_initial
        else:
            p_t = self.p_left_initial * np.exp(-self.time_decay * t)
            # Ensure probability stays within [0, 1]
            return max(0.0, min(p_t, 1.0))

    def simulate_trajectory(self, start_state: int, n_steps: int) -> list:
        """Simulate a trajectory for the random walk."""
        if not self.state_min <= start_state <= self.state_max:
            start_state = max(self.state_min, min(self.state_max, start_state))
            st.warning(f"Start state clamped to {start_state}...")

        trajectory = [start_state]
        current_state = start_state

        for t in range(n_steps):
            # Get p_left for the current time step t
            current_p_left = self.get_p_left(t)

            if np.random.rand() < current_p_left:
                next_state = current_state - 1
            else:
                next_state = current_state + 1

            current_state = max(self.state_min, min(self.state_max, next_state))
            trajectory.append(current_state)

        return trajectory


def create_animation(trajectory, visible_N, animation_speed_ms):
    fig, ax = plt.subplots(figsize=(12, 3))

    # Determine dynamic window based on trajectory range but keep it centered if possible
    start_pos = trajectory[0]
    min_traj = min(trajectory)
    max_traj = max(trajectory)
    center = start_pos  # Keep window centered around start unless trajectory goes far

    # Adjust window if trajectory goes beyond the initial visible_N range
    window_half_width = visible_N // 2
    window_min_calc = min(center - window_half_width, min_traj - 2)
    window_max_calc = max(center + window_half_width, max_traj + 2)

    # Ensure window size is at least visible_N
    current_width = window_max_calc - window_min_calc
    if current_width < visible_N:
        diff = visible_N - current_width
        window_min_calc -= diff // 2
        window_max_calc += (diff + 1) // 2  # Add 1 for odd diffs

    ax.set_xlim(window_min_calc, window_max_calc)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])  # Hide y-axis ticks

    # Draw the line segment with dots indicating continuation
    ax.plot([window_min_calc, window_max_calc], [0, 0], "k-", lw=2)
    ax.plot([window_min_calc], [0], "k.", markersize=10)  # Dot at the left
    ax.plot([window_max_calc], [0], "k.", markersize=10)  # Dot at the right

    # Add grid for better position visibility
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer ticks
    ax.grid(True, axis="x", linestyle=":", alpha=0.7)

    # Create marker for the agent and its trail
    (agent,) = ax.plot([], [], "ro", markersize=12, label="Agent")
    (trail,) = ax.plot([], [], "r--", alpha=0.5, lw=1, label="Trail")  # Dashed trail

    # Text to display current step and position
    time_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    pos_text = ax.text(0.85, 0.90, "", transform=ax.transAxes)

    def init():
        agent.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        pos_text.set_text("")
        return agent, trail, time_text, pos_text

    def animate(i):
        current_pos = trajectory[i]
        agent.set_data([current_pos], [0])
        # Update trail data up to current frame
        trail.set_data(trajectory[: i + 1], np.zeros(i + 1))
        time_text.set_text(f"Step: {i}")
        pos_text.set_text(f"Position: {current_pos}")
        return agent, trail, time_text, pos_text

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(trajectory),
        interval=animation_speed_ms,  # Use parameter here
        blit=True,
        repeat=False,
    )

    # Revert to simple save and return path
    gif_path = "animation.gif"
    try:
        writer = PillowWriter(fps=int(1000 / animation_speed_ms))
        anim.save(gif_path, writer=writer)
        plt.close(fig)  # Close figure after saving
        return fig, anim, gif_path  # Return path
    except Exception as e:
        st.error(f"Could not save animation: {e}")
        plt.close(fig)
        return fig, anim, None  # Return None for path if failed


# --- MDP Environment Definition (Based on Image) ---
class RichFamousMDP:
    def __init__(self):
        self.states = ["poor, unknown", "poor, famous", "rich, unknown", "rich, famous"]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.actions = ["Save (S)", "Advertise (A)"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)

        # Define Transitions P(s'|s, a) and Rewards R(s, a, s')
        # P[action_idx, state_idx, next_state_idx]
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        # R[action_idx, state_idx, next_state_idx] - Note: image shows reward R(s,a)
        # We can average R(s,a,s') over s' to get R(s,a) = sum_{s'} P(s'|s,a)R(s,a,s')
        # Or, if reward depends just on (s,a), we store R[action, state]
        self.R = np.zeros((self.num_actions, self.num_states))

        # State Indices:
        PU = self.state_to_idx["poor, unknown"]
        PF = self.state_to_idx["poor, famous"]
        RU = self.state_to_idx["rich, unknown"]
        RF = self.state_to_idx["rich, famous"]

        # Actions Indices:
        S = self.action_to_idx["Save (S)"]
        A = self.action_to_idx["Advertise (A)"]

        # --- Transitions and Rewards from Image ---
        # Action: Save (S)
        # From poor, unknown (PU)
        self.P[S, PU, PU] = 1.0
        self.R[S, PU] = 0  # R(s=PU, a=S) = 0
        # From poor, famous (PF)
        self.P[S, PF, RU] = 0.5
        self.P[S, PF, RF] = 0.5
        self.R[S, PF] = 0  # R(s=PF, a=S) = 0
        # From rich, unknown (RU)
        self.P[S, RU, PU] = 0.5
        self.P[S, RU, RU] = 0.5
        self.R[S, RU] = 10  # R(s=RU, a=S) = 10
        # From rich, famous (RF)
        self.P[S, RF, RF] = 0.5
        self.P[S, RF, RU] = 0.5  # Typo in diagram? Assuming save can make you unknown
        self.R[S, RF] = 10  # R(s=RF, a=S) = 10

        # Action: Advertise (A)
        # From poor, unknown (PU)
        self.P[A, PU, PU] = 0.5
        self.P[A, PU, PF] = 0.5
        self.R[A, PU] = -1  # R(s=PU, a=A) = -1
        # From poor, famous (PF)
        self.P[A, PF, PF] = 1.0
        self.R[A, PF] = -1  # R(s=PF, a=A) = -1
        # From rich, unknown (RU)
        self.P[A, RU, RF] = 1.0
        self.R[A, RU] = -1  # R(s=RU, a=A) = -1
        # From rich, famous (RF)
        self.P[A, RF, PF] = 1.0
        self.R[A, RF] = -1  # R(s=RF, a=A) = -1

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_transitions(self):
        return self.P

    def get_rewards(self):
        # Returns R[action, state] as derived from image R(s,a)
        return self.R


# --- MDP Policy Definitions ---
def policy_always_save(state_idx, mdp):
    return mdp.action_to_idx["Save (S)"]


def policy_always_advertise(state_idx, mdp):
    return mdp.action_to_idx["Advertise (A)"]


def policy_mixed_simple(state_idx, mdp):
    """Simple policy: Save if rich, Advertise if poor."""
    state_name = mdp.idx_to_state[state_idx]
    if "rich" in state_name:
        return mdp.action_to_idx["Save (S)"]
    else:
        return mdp.action_to_idx["Advertise (A)"]


def policy_advertise_if_unknown(state_idx, mdp):
    """Advertise only if unknown, otherwise save."""
    state_name = mdp.idx_to_state[state_idx]
    if "unknown" in state_name:
        return mdp.action_to_idx["Advertise (A)"]
    else:
        return mdp.action_to_idx["Save (S)"]


def policy_random(state_idx, mdp):
    """Choose an action randomly."""
    return np.random.choice(mdp.num_actions)


# Dictionary to hold policy functions
mdp_policies = {
    "Always Save": policy_always_save,
    "Always Advertise": policy_always_advertise,
    "Mixed (Save if Rich, Adv. if Poor)": policy_mixed_simple,
    "Advertise if Unknown": policy_advertise_if_unknown,
    "Random": policy_random,
}


# --- MDP Trajectory Simulation ---
def simulate_mdp_trajectory(mdp, policy_func, start_state_name, n_steps):
    """Simulate a single trajectory in the MDP given a policy."""
    if start_state_name not in mdp.state_to_idx:
        st.error(f"Invalid start state: {start_state_name}")
        return None, None

    start_state_idx = mdp.state_to_idx[start_state_name]
    trajectory_states = [start_state_idx]
    trajectory_rewards = [0]  # Reward received *before* this state (or 0 for start)
    total_reward = 0
    current_state_idx = start_state_idx
    P = mdp.get_transitions()
    R = mdp.get_rewards()  # R[action, state]

    for _ in range(n_steps):
        # Get action from policy
        action_idx = policy_func(current_state_idx, mdp)

        # Get reward for taking action a in state s
        reward = R[action_idx, current_state_idx]
        total_reward += reward  # Using R(s,a) - immediate reward

        # Get next state probabilities
        next_state_probs = P[action_idx, current_state_idx, :]

        # Sample next state
        if np.sum(next_state_probs) > 0:  # Avoid errors if no transitions defined
            next_state_idx = np.random.choice(mdp.num_states, p=next_state_probs)
        else:
            # If no transitions, assume stay in same state (or handle error)
            next_state_idx = current_state_idx
            st.warning(
                f"No transitions defined for state {mdp.idx_to_state[current_state_idx]}, action {mdp.actions[action_idx]}. Staying put."
            )

        trajectory_states.append(next_state_idx)
        trajectory_rewards.append(
            reward
        )  # Store reward received *before* reaching next state
        current_state_idx = next_state_idx

    return trajectory_states, trajectory_rewards, total_reward


# --- Function to calculate Induced Markov Chain ---
def calculate_induced_markov_chain(mdp, policy_func):
    """Calculates the transition matrix P_pi for the Markov chain induced by a policy.

    Args:
        mdp: The RichFamousMDP instance.
        policy_func: The policy function (deterministic: state_idx -> action_idx).

    Returns:
        P_pi: An (N_states x N_states) numpy array representing p^pi(s'|s).
    """
    P_pi = np.zeros((mdp.num_states, mdp.num_states))
    P_orig = mdp.get_transitions()  # P[action, state, next_state]

    for s_idx in range(mdp.num_states):
        # Get the single action dictated by the deterministic policy
        action_idx = policy_func(s_idx, mdp)

        # The transition probabilities for this state are just the ones for the chosen action
        P_pi[s_idx, :] = P_orig[action_idx, s_idx, :]

        # --- Note for future stochastic policies ---
        # If policy was stochastic (policy_func returned pi(a|s) for all a),
        # the calculation would be:
        # for s_prime_idx in range(mdp.num_states):
        #     prob = 0
        #     for a_idx in range(mdp.num_actions):
        #         pi_a_s = get_pi(policy_func, s_idx, a_idx, mdp) # Assume function to get pi(a|s)
        #         prob += pi_a_s * P_orig[a_idx, s_idx, s_prime_idx]
        #     P_pi[s_idx, s_prime_idx] = prob
        # --- End Note ---

    return P_pi


# --- Instantiate MDP early ---
# Create the MDP instance here so it's available for both tabs
mdp = RichFamousMDP()

# Streamlit app
st.set_page_config(layout="wide")
st.title("Markov Process and MDP Simulation")

# --- Main Area with Tabs ---
# No sidebar, controls are inside tabs now.

tab1, tab2 = st.tabs(["Simple Random Walk (Markov Process)", "Rich/Famous MDP Example"])

# --- Tab 1: Markov Process ---
with tab1:
    # Layout: Controls on Left (width 1), Output on Right (width 3)
    col1_controls, col1_output = st.columns([1, 3])

    # --- Controls Column (Tab 1) ---
    with col1_controls:
        # Revert to st.container with a fixed height for scrolling
        # Remove the div wrapper
        with st.container(height=700):
            st.header("MP Settings")

            st.subheader("General Visualization")
            visible_N = st.slider(
                "Visible Line Width",
                10,
                50,
                20,
                help="How many positions to show visually for the line process.",
                key="mp_viz_width",
            )
            animation_speed_ms = st.slider(
                "Line Animation Speed (ms per step)",
                100,
                1000,
                500,
                100,
                key="mp_viz_speed",
            )

            st.subheader("Process Parameters")
            start_state_line = st.number_input(
                "Start Position", value=0, key="mp_start"
            )
            n_steps_line = st.slider("Number of Steps", 10, 200, 50, key="mp_steps")
            process_type_line = st.selectbox(
                "Process Type",
                ["Time Homogeneous", "Time Non-homogeneous"],
                key="line_process_type",
            )
            p_left_initial = st.slider(
                "Initial Probability Move Left (p₀)",
                0.0,
                1.0,
                0.5,
                0.05,
                key="line_p_left_initial",
                help="Probability of moving left at t=0.",
            )
            time_decay_line = 0.0  # Initialize here
            if process_type_line == "Time Non-homogeneous":
                time_decay_line = st.slider(
                    "Time Decay Factor (λ)",
                    0.0,
                    0.5,
                    0.05,
                    0.01,
                    key="line_time_decay",
                    help="Rate at which p_left decays: p_left(t) = p₀ * exp(-λ*t)",
                )

            st.subheader("Batch Simulation")
            num_simulations = st.slider(
                "Number of Walks (N)", 10, 1000, 100, 10, key="mp_num_sims"
            )

    # --- Output Column (Tab 1) ---
    with col1_output:
        st.header("Simple Random Walk on a Line (Markov Process)")
        st.write("Visualize a 1D random walk.")

        # Determine homogeneity based on control
        is_homogeneous_line = process_type_line == "Time Homogeneous"

        st.subheader("Process Dynamics")
        if is_homogeneous_line:
            st.write(
                rf"The process is time-homogeneous. Probability of moving left is constant: \(p = p_0 = {p_left_initial:.2f}\). Probability right is \(1-p = {1-p_left_initial:.2f}\)."
            )
            st.latex(rf"P(X_{{t+1}} = i-1 \mid X_t = i) = p = {p_left_initial:.2f}")
            st.latex(rf"P(X_{{t+1}} = i+1 \mid X_t = i) = 1-p = {1-p_left_initial:.2f}")
        else:
            st.write(
                r"The process is time non-homogeneous. Probability of moving left \(p_t\) decays over time:"
            )
            st.latex(
                rf"p_t = p_0 \times e^{{-\lambda t}} = {p_left_initial:.2f} \times e^{{-{time_decay_line:.2f} t}}"
            )
            st.write(r"Probability right is \(1-p_t\). The transitions are:")
            st.latex(r"P(X_{{t+1}} = i-1 \mid X_t = i) = p_t")
            st.latex(r"P(X_{{t+1}} = i+1 \mid X_t = i) = 1-p_t")

        st.subheader("Run a Single Simulation")
        if st.button("Run Single Simulation", key="run_single_walk_button"):
            env = LineEnvironment(
                p_left_initial=p_left_initial,
                time_homogeneous=is_homogeneous_line,
                time_decay=time_decay_line,
            )
            st.session_state.single_trajectory = env.simulate_trajectory(
                start_state_line, n_steps_line
            )
            st.success(
                f"Generated a single random walk of {n_steps_line} steps starting at {start_state_line}."
            )
            # Clear potentially conflicting results
            st.session_state.pop("animation_gif_path", None)
            st.session_state.pop("batch_final_states", None)
            st.session_state.pop("batch_states_at_time", None)

        if "single_trajectory" in st.session_state:
            st.markdown("--- ")
            st.subheader("View Single Trajectory Animation")
            if st.button("Generate & Show Animation", key="show_walk_anim_button"):
                st.session_state.pop(
                    "animation_gif_path", None
                )  # Clear previous before generating
                with st.spinner("Generating animation..."):
                    trajectory_to_show = st.session_state.single_trajectory
                    fig_anim, anim, gif_path = create_animation(
                        trajectory_to_show, visible_N, animation_speed_ms
                    )
                    if gif_path:
                        st.session_state.animation_gif_path = gif_path
                    else:
                        st.warning("Animation generation failed.")

            if "animation_gif_path" in st.session_state:
                gif_path_to_display = st.session_state.animation_gif_path
                try:
                    st.image(gif_path_to_display)
                    with open(gif_path_to_display, "rb") as file:
                        st.download_button(
                            label="Download Animation (GIF)",
                            data=file,
                            file_name="random_walk_animation.gif",
                            mime="image/gif",
                            key="download_walk_gif_button",
                        )
                except FileNotFoundError:
                    st.error("Animation file not found. Please regenerate.")
                except Exception as e_disp:
                    st.error(f"Error displaying animation: {e_disp}")
            st.markdown("--- ")

        st.subheader("Analyze Multiple Random Walks")
        if st.button("Run N Walks", key="run_batch_walk_button"):
            st.write(f"Running {num_simulations} random walks...")
            progress_bar = st.progress(0)
            all_final_states = []
            time_milestones = set([0, n_steps_line])
            if n_steps_line >= 1:
                time_milestones.add(1)
            if n_steps_line >= 2:
                time_milestones.add(2)
            if n_steps_line >= 4:
                time_milestones.add(n_steps_line // 4)
                time_milestones.add(n_steps_line // 2)
                time_milestones.add((3 * n_steps_line) // 4)
            time_milestones = sorted(list(time_milestones)) or [0]
            states_at_time = {t: [] for t in time_milestones}

            env = LineEnvironment(
                p_left_initial=p_left_initial,
                time_homogeneous=is_homogeneous_line,
                time_decay=time_decay_line,
            )
            for i in range(num_simulations):
                trajectory = env.simulate_trajectory(start_state_line, n_steps_line)
                final_state = trajectory[-1] if trajectory else start_state_line
                all_final_states.append(final_state)
                for t in states_at_time:
                    if t < len(trajectory):
                        states_at_time[t].append(trajectory[t])
                progress_bar.progress((i + 1) / num_simulations)

            st.success(f"Completed {num_simulations} walks.")
            progress_bar.empty()
            st.session_state.batch_final_states = all_final_states
            st.session_state.batch_states_at_time = states_at_time
            # Clear potentially conflicting results
            st.session_state.pop("single_trajectory", None)
            st.session_state.pop("animation_gif_path", None)

        if "batch_final_states" in st.session_state:
            st.markdown("### Batch Simulation Results")
            final_states = st.session_state.batch_final_states
            states_at_time_data = st.session_state.batch_states_at_time

            st.markdown(f"#### Distribution of Final Positions (t={n_steps_line})")
            fig_hist_final = plt.figure(figsize=(10, 4))
            if final_states:
                bin_count = max(
                    10, int(np.ptp(final_states) if np.ptp(final_states) > 0 else 1) + 1
                )
                plt.hist(final_states, bins=bin_count, density=True, alpha=0.7)
                plt.title(
                    f"Distribution after {n_steps_line} Steps (N={len(final_states)})"
                )
                plt.xlabel(f"Final Position (s_{{{n_steps_line}}})")
                plt.ylabel("Frequency")
            else:
                plt.title(f"Distribution of Final Position (No data)")
            st.pyplot(fig_hist_final)
            plt.close(fig_hist_final)

            st.markdown("#### Distribution of Positions Over Time")
            times = sorted(states_at_time_data.keys())
            if times:  # Ensure there are times to plot
                fig_dist_time, axes = plt.subplots(
                    1, len(times), figsize=(max(8, 3.5 * len(times)), 4), sharey=True
                )
                if len(times) == 1:
                    axes = [axes]
                for i, t in enumerate(times):
                    states = states_at_time_data[t]
                    ax = axes[i]
                    if states:
                        mean_pos = np.mean(states)
                        std_dev = np.std(states)
                        if np.isclose(std_dev, 0):
                            hist_range = (mean_pos - 1.5, mean_pos + 1.5)
                            bins = 3
                        else:
                            min_s, max_s = np.min(states), np.max(states)
                            hist_range = (min_s - 1, max_s + 1)
                            bins = max(5, int(max_s - min_s) + 1)
                        ax.hist(
                            states, bins=bins, density=True, alpha=0.7, range=hist_range
                        )
                        ax.set_title(f"Position Dist. at t={t}")
                        ax.set_xlabel(f"Position (s_{{{t}}})")
                        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    else:
                        ax.set_title(f"Position Dist. at t={t} (No data)")
                        ax.set_xlabel(f"Position (s_{{{t}}})")
                axes[0].set_ylabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_dist_time)
                plt.close(fig_dist_time)
            else:
                st.write("(No time distribution data)")

            st.markdown("#### Summary Statistics from Batch Run")
            col1, col2, col3 = st.columns(3)
            num_walks_run = len(final_states)
            with col1:
                st.metric(
                    "Avg. Final Position",
                    f"{np.mean(final_states):.2f}" if final_states else "N/A",
                )
            with col2:
                st.metric(
                    "Std Dev Final Position",
                    f"{np.std(final_states):.2f}" if final_states else "N/A",
                )
            with col3:
                st.metric("Walks Run (N)", num_walks_run)

# --- Tab 2: MDP Implementation (Policy Simulation) ---
with tab2:
    # Layout: Controls on Left (width 1), Output on Right (width 3)
    col2_controls, col2_output = st.columns([1, 3])

    # --- Controls Column (Tab 2) ---
    with col2_controls:
        # Revert to st.container with a fixed height for scrolling
        # Remove the div wrapper
        with st.container(height=700):
            st.header("MDP Settings")

            st.subheader("Policy & Simulation")
            selected_policy_name = st.selectbox(
                "Select Policy", list(mdp_policies.keys()), key="mdp_policy_select"
            )
            start_state_mdp = st.selectbox(
                "Select Start State", mdp.states, key="mdp_start_state"
            )
            n_steps_mdp = st.slider(
                "Number of Steps (T)", 1, 100, 20, key="mdp_n_steps"
            )

            st.subheader("Batch Simulation")
            num_simulations_mdp = st.slider(
                "Number of Simulations (N)", 10, 1000, 100, 10, key="mdp_num_sims"
            )

            st.subheader("Policy Comparison")
            num_compare_sims = st.slider(
                "Sims per Policy (N_compare)",
                50,
                2000,
                200,
                50,
                key="mdp_compare_num_sims",
            )
            n_steps_compare = st.slider(
                "Steps per Sim (T_compare)", 10, 200, 50, 10, key="mdp_compare_n_steps"
            )

    # --- Output Column (Tab 2) ---
    with col2_output:
        st.header("Rich/Famous MDP Example - Policy Simulation")
        st.caption(
            "MDP Example Source: Probabilistic Artificial Intelligence, Andreas Krause, Jonas Hübotter"
        )
        st.write("Simulate trajectories under different fixed policies.")
        try:
            st.image("mdp_diagram.png", caption="MDP Diagram", width=500)
        except FileNotFoundError:
            st.warning("MDP diagram image ('mdp_diagram.png') not found.")
        except Exception as e:
            st.error(f"Error loading MDP diagram: {e}")

        st.subheader("MDP Definition")
        with st.expander("Show States, Actions, Transitions, Rewards"):
            st.markdown("**States:**")
            st.json({i: s for i, s in mdp.idx_to_state.items()})
            st.markdown("**Actions:**")
            st.json({i: a for i, a in enumerate(mdp.actions)})
            st.markdown("**Transition Probabilities P(s'|s, a):**")
            for i, action_name in enumerate(mdp.actions):
                st.markdown(f"*Action: {action_name}* (P[{i}, state, next_state])")
                df_P = pd.DataFrame(mdp.P[i], index=mdp.states, columns=mdp.states)
                st.dataframe(df_P)
            st.markdown("**Rewards R(s, a):**")
            rewards_dict = {}
            for i, action_name in enumerate(mdp.actions):
                rewards_dict[action_name] = {
                    state_name: mdp.R[i, s_idx]
                    for s_idx, state_name in mdp.idx_to_state.items()
                }
            st.json(rewards_dict)

        st.subheader("Policy Simulation Settings")
        # Display the selected settings from controls column
        st.write(f"**Selected Policy:** {selected_policy_name}")
        st.write(f"**Selected Start State:** {start_state_mdp}")
        st.write(f"**Simulation Steps (T):** {n_steps_mdp}")

        policy_func = mdp_policies[selected_policy_name]

        st.markdown("--- ")
        st.subheader("Induced Markov Chain for Selected Policy")
        st.latex(
            r"p^\pi(x' \mid x) = \sum_{a \in \mathcal{A}} \pi(a \mid x) \cdot p(x' \mid x, a)"
        )
        st.markdown(
            r"""
        When we fix a policy $\pi(a|x)$ (which tells us the probability of choosing action $a$ in state $x$),
        the transitions between states no longer depend on *choosing* an action. Instead, they depend only on the
        current state $x$ and the fixed probabilities defined by the policy.

        This means the system behaves exactly like a standard Markov chain over the state space $\mathcal{S}$.
        The transition probabilities for this *induced* Markov chain, denoted $p^\pi(x' \mid x)$, are calculated by
        averaging the original MDP transitions $p(x' \mid x, a)$ according to the policy's action probabilities $\pi(a \mid x)$.
        (Since our current policies are deterministic, $\pi(a|x)=1$ for the chosen action and 0 otherwise, simplifying the sum.)
        """
        )
        P_pi = calculate_induced_markov_chain(mdp, policy_func)
        df_P_pi = pd.DataFrame(
            P_pi,
            index=mdp.states,
            columns=mdp.states,
        )
        st.markdown(
            rf"**Induced Markov Chain Transition Matrix $P^\pi$ for: {selected_policy_name}**"
        )
        st.dataframe(df_P_pi.round(3))
        st.markdown(
            r"Notice that this $P^\pi$ matrix defines transitions based *only* on the current state $x$ and the next state $x'$."
        )
        st.markdown("--- ")

        st.subheader("Run Single Simulation (Under Selected Policy)")
        if st.button("Run Single MDP Simulation", key="run_single_mdp_button"):
            traj_states, traj_rewards, total_reward = simulate_mdp_trajectory(
                mdp, policy_func, start_state_mdp, n_steps_mdp
            )
            if traj_states:
                st.session_state.mdp_single_trajectory_states = [
                    mdp.idx_to_state[s] for s in traj_states
                ]
                st.session_state.mdp_single_trajectory_rewards = traj_rewards
                st.session_state.mdp_single_total_reward = total_reward
                st.success(f"Generated single MDP trajectory of {n_steps_mdp} steps.")
                # Clear conflicting results
                st.session_state.pop("mdp_batch_final_states", None)
                st.session_state.pop("mdp_batch_states_at_time", None)
                st.session_state.pop("mdp_batch_avg_reward", None)

        if "mdp_single_trajectory_states" in st.session_state:
            st.markdown("**Single Trajectory:**")
            st.write(" -> ".join(st.session_state.mdp_single_trajectory_states))
            st.metric(
                "Total Reward for this Trajectory",
                f"{st.session_state.mdp_single_total_reward:.2f}",
            )
            st.caption(
                "Note: Total Reward is the undiscounted sum of immediate rewards received over T steps."
            )
            st.markdown("--- ")

        st.subheader("Analyze Multiple Simulations (Under Selected Policy)")
        if st.button("Run N MDP Simulations", key="run_batch_mdp_button"):
            st.write(
                f"Running {num_simulations_mdp} simulations under policy: {selected_policy_name}..."
            )
            progress_bar_mdp = st.progress(0)
            all_final_states_mdp = []
            all_total_rewards_mdp = []
            time_milestones_mdp = set([0, n_steps_mdp])
            if n_steps_mdp >= 1:
                time_milestones_mdp.add(1)
            if n_steps_mdp >= 2:
                time_milestones_mdp.add(2)
            if n_steps_mdp >= 4:
                time_milestones_mdp.add(n_steps_mdp // 4)
                time_milestones_mdp.add(n_steps_mdp // 2)
                time_milestones_mdp.add((3 * n_steps_mdp) // 4)
            time_milestones_mdp = sorted(list(time_milestones_mdp)) or [0]
            states_at_time_mdp = {t: [] for t in time_milestones_mdp}

            for i in range(num_simulations_mdp):
                traj_states, _, total_reward = simulate_mdp_trajectory(
                    mdp, policy_func, start_state_mdp, n_steps_mdp
                )
                if traj_states:
                    final_state_name = mdp.idx_to_state[traj_states[-1]]
                    all_final_states_mdp.append(final_state_name)
                    all_total_rewards_mdp.append(total_reward)
                    for t in states_at_time_mdp:
                        if t < len(traj_states):
                            states_at_time_mdp[t].append(
                                mdp.idx_to_state[traj_states[t]]
                            )
                progress_bar_mdp.progress((i + 1) / num_simulations_mdp)

            st.success(f"Completed {num_simulations_mdp} MDP simulations.")
            progress_bar_mdp.empty()
            st.session_state.mdp_batch_final_states = all_final_states_mdp
            st.session_state.mdp_batch_states_at_time = states_at_time_mdp
            st.session_state.mdp_batch_avg_reward = (
                np.mean(all_total_rewards_mdp) if all_total_rewards_mdp else 0
            )
            # Clear conflicting results
            st.session_state.pop("mdp_single_trajectory_states", None)
            st.session_state.pop("mdp_single_trajectory_rewards", None)
            st.session_state.pop("mdp_single_total_reward", None)

        if "mdp_batch_final_states" in st.session_state:
            st.markdown("### Batch Simulation Results")
            final_states = st.session_state.mdp_batch_final_states
            states_at_time_data = st.session_state.mdp_batch_states_at_time
            avg_reward = st.session_state.mdp_batch_avg_reward

            st.metric("Average Total Reward over N runs", f"{avg_reward:.3f}")

            st.markdown(f"#### Distribution of Final States (t={n_steps_mdp})")
            fig_hist_final_mdp = plt.figure(figsize=(8, 4))
            if final_states:
                state_counts = (
                    pd.Series(final_states)
                    .value_counts()
                    .reindex(mdp.states, fill_value=0)
                )
                plt.bar(
                    state_counts.index,
                    state_counts.values / len(final_states),
                    alpha=0.7,
                )
                plt.title(
                    f"Distribution of State after {n_steps_mdp} Steps (N={len(final_states)})"
                )
                plt.ylabel("Frequency")
                plt.xticks(rotation=45, ha="right")
            else:
                plt.title(f"Distribution of Final State (No data)")
            st.pyplot(fig_hist_final_mdp)
            plt.close(fig_hist_final_mdp)

            st.markdown("#### Distribution of States Over Time")
            times_mdp = sorted(states_at_time_data.keys())
            if times_mdp:
                fig_dist_time_mdp, axes_mdp = plt.subplots(
                    1,
                    len(times_mdp),
                    figsize=(max(8, 3.5 * len(times_mdp)), 4),
                    sharey=True,
                )
                if len(times_mdp) == 1:
                    axes_mdp = [axes_mdp]
                for i, t in enumerate(times_mdp):
                    states = states_at_time_data[t]
                    ax = axes_mdp[i]
                    if states:
                        state_counts_t = (
                            pd.Series(states)
                            .value_counts()
                            .reindex(mdp.states, fill_value=0)
                        )
                        ax.bar(
                            state_counts_t.index,
                            state_counts_t.values / len(states),
                            alpha=0.7,
                        )
                        ax.set_title(f"State Dist. at t={t}")
                        ax.tick_params(axis="x", rotation=45)
                    else:
                        ax.set_title(f"State Dist. at t={t} (No data)")
                axes_mdp[0].set_ylabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_dist_time_mdp)
                plt.close(fig_dist_time_mdp)
            else:
                st.write("(No time distribution data)")

        st.markdown("--- ")
        st.subheader("Compare Average Rewards Across Policies")
        st.write(
            "Run batch simulations for all defined policies starting from 'poor, unknown' to compare their average performance."
        )

        start_state_compare = "poor, unknown"
        st.write(
            f"Comparison Settings: N={num_compare_sims}, T={n_steps_compare}, Start State='{start_state_compare}'"
        )

        if st.button("Run Policy Comparison", key="run_policy_compare_button"):
            st.write(
                f"Comparing policies over {num_compare_sims} runs of {n_steps_compare} steps each, starting from '{start_state_compare}'..."
            )
            policy_comparison_results = {}
            comparison_progress = st.progress(0)
            policy_names = list(mdp_policies.keys())
            for i, policy_name in enumerate(policy_names):
                policy_to_run = mdp_policies[policy_name]
                policy_rewards = []
                for _ in range(num_compare_sims):
                    _, _, total_reward = simulate_mdp_trajectory(
                        mdp, policy_to_run, start_state_compare, n_steps_compare
                    )
                    if total_reward is not None:
                        policy_rewards.append(total_reward)
                policy_comparison_results[policy_name] = (
                    np.mean(policy_rewards) if policy_rewards else None
                )
                comparison_progress.progress((i + 1) / len(policy_names))
            st.success("Policy comparison complete!")
            comparison_progress.empty()
            st.session_state.policy_comparison_results = policy_comparison_results

        if "policy_comparison_results" in st.session_state:
            st.markdown("**Average Total Reward Comparison:**")
            comparison_data = st.session_state.policy_comparison_results
            sorted_comparison = sorted(
                comparison_data.items(),
                key=lambda item: item[1] if item[1] is not None else -np.inf,
                reverse=True,
            )
            df_comparison = pd.DataFrame(
                sorted_comparison, columns=["Policy", "Average Total Reward"]
            )
            st.dataframe(df_comparison.round(3))

# Cleaned up old logic previously - Removed sidebar and default value logic
