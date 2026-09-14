import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import gymnasium as gym
from tqdm import trange
from IPython.display import clear_output
import copy
import json
from pathlib import Path


def current_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
current_device() #to display after importing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_global_iterations_instants_temporal_horizon(system_dynamics_params_, horizon):
    # ... (unchanged - remains on CPU as it uses numpy)
    requests_instants = []
    last_instant = np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    while last_instant < horizon:
        requests_instants.append(last_instant)
        last_instant += np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    mark_weights = np.sort(np.random.rand(system_dynamics_params_['n_classes']))
    scaled_weights = np.exp(mark_weights * system_dynamics_params_['gamma_heter'])
    mark_probabilities = scaled_weights / np.sum(scaled_weights)
    marks = np.random.choice(np.arange(4), p=mark_probabilities, size=len(requests_instants))
    global_iterations_instants = [[requests_instants[i] for i in range(len(marks)) if marks[i] == c]
                                  for c in range(system_dynamics_params_["n_classes"])]
    oracle_iterations = []
    indices = np.zeros(system_dynamics_params_['n_classes'], dtype=int)
    for m in marks:
        oracle_iterations.append((int(m), int(indices[m])))
        indices[m] += 1
    return global_iterations_instants, oracle_iterations



def generate_global_iterations_instants_holistic(system_dynamics_params_, horizon):
    # ... (unchanged)
    requests_instants = []
    last_instant = np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    for _ in range(horizon):
        requests_instants.append(last_instant)
        last_instant += np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    mark_weights = np.sort(np.random.rand(system_dynamics_params_['n_classes']))
    scaled_weights = np.exp(mark_weights * system_dynamics_params_['gamma_heter'])
    mark_probabilities = scaled_weights / np.sum(scaled_weights)
    marks = np.random.choice(np.arange(4), p=mark_probabilities, size=len(requests_instants))
    global_iterations_instants = [[requests_instants[i] for i in range(len(marks)) if marks[i] == c]
                                  for c in range(system_dynamics_params_["n_classes"])]
    oracle_iterations = []
    indices = np.zeros(system_dynamics_params_['n_classes'], dtype=int)
    for m in marks:
        oracle_iterations.append((int(m), int(indices[m])))
        indices[m] += 1
    return global_iterations_instants, oracle_iterations, requests_instants[-1]



def generate_global_iterations_instants_reductionist(system_dynamics_params_, horizon):
    # ... (unchanged)
    requests_instants = []
    last_instant = np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    mark_weights = np.sort(np.random.rand(system_dynamics_params_['n_classes']))
    scaled_weights = np.exp(mark_weights * system_dynamics_params_['gamma_heter'])
    mark_probabilities = scaled_weights / np.sum(scaled_weights)
    global_iterations_instants = [[] for _ in range(system_dynamics_params_["n_classes"])]
    oracle_iterations = []
    indices = np.zeros(system_dynamics_params_['n_classes'], dtype=int)
    while sum(indices >= np.repeat(horizon, system_dynamics_params_['n_classes'])) < system_dynamics_params_['n_classes']:
        requests_instants.append(last_instant)
        class_ind = np.random.choice(np.arange(4), p=mark_probabilities)
        global_iterations_instants[class_ind].append(last_instant)
        oracle_iterations.append((int(class_ind), int(indices[class_ind])))
        indices[class_ind] += 1
        last_instant += np.random.exponential(np.random.exponential(system_dynamics_params_['gamma_req']))
    return global_iterations_instants, oracle_iterations, requests_instants[-1]



def generate_learning_periods(system_dynamics_params_, horizon):
    # ... (unchanged)
    learning_periods = [[] for i in range(system_dynamics_params_["n_octopi"])]
    for i in range(system_dynamics_params_["n_octopi"]):
        period = (0, np.random.exponential(system_dynamics_params_['gamma_learn']))
        while period[-1] <= horizon:
            learning_periods[i].append(period)
            break_duration = np.random.exponential(system_dynamics_params_['gamma_break'])
            period = (period[1] + break_duration,
                      period[1] + break_duration + np.random.exponential(system_dynamics_params_['gamma_learn']))

    u_instants = np.concatenate([[p[-1] for p in learning_periods[i]]
                                 for i in range(system_dynamics_params_['n_octopi'])] + [[0]])
    u_instants.sort()
    return learning_periods, u_instants



def generate_triggering_events_instants(system_dynamics_params_, global_iterations_instants, activation_periods):
    # ... (unchanged)
    triggering_events_instants = [[[] for c in range(system_dynamics_params_['n_classes'])]
                                  for i in range(system_dynamics_params_['n_octopi'])]
    number_of_episodes = [[[] for c in range(system_dynamics_params_['n_classes'])]
                          for i in range(system_dynamics_params_['n_octopi'])]

    for c in range(system_dynamics_params_['n_classes']):
        for y in global_iterations_instants[c]:
            for i in range(system_dynamics_params_['n_octopi']):
                if not activation_periods[i]:
                    continue
                period_ind = len([period for period in activation_periods[i] if period[0] < y]) - 1
                if period_ind < 0:
                    continue
                if y < activation_periods[i][period_ind][1]:
                    if np.random.rand() <= system_dynamics_params_['gamma_sync']:
                        z = y
                        triggering_events_instants[i][c].append(z)
                        training_period = z - activation_periods[i][period_ind][0]
                        if len(triggering_events_instants[i][c]) > 1:
                            last_event_instant = max(triggering_events_instants[i][c][-2],
                                                     activation_periods[i][period_ind][0])
                            training_period = triggering_events_instants[i][c][-1] - last_event_instant
                        n_episodes = np.random.poisson(system_dynamics_params_['gamma_epis'] * training_period)
                        number_of_episodes[i][c].append(n_episodes)

    return triggering_events_instants, number_of_episodes


def generate_instants(system_dynamics_params_, horizon):
    # ... (unchanged)
    global_iterations_instants, oracle_iterations, final_time = generate_global_iterations_instants_holistic(
        system_dynamics_params_, horizon)
    activation_periods, u_instants = generate_learning_periods(system_dynamics_params_, horizon)
    triggering_events_instants, number_of_episodes = generate_triggering_events_instants(
        system_dynamics_params_, global_iterations_instants, activation_periods)

    return global_iterations_instants, activation_periods, u_instants, triggering_events_instants, number_of_episodes, oracle_iterations


# ====================== REMAINING HELPER FUNCTIONS (unchanged) ======================
def activations_candidacies(activation_periods, triggering_events, global_iterations_instants):
    # ... (unchanged - pure numpy logic)
    n_classes = len(global_iterations_instants)
    Activations = [np.zeros(len(global_iterations_instants[class_ind])) for class_ind in range(n_classes)]
    Candidacies = [np.zeros(len(global_iterations_instants[class_ind])) for class_ind in range(n_classes)]
    Last_triggering_events_indices = np.zeros(n_classes, dtype=int)

    if not activation_periods:
        return Activations, Candidacies

    for class_ind in range(n_classes):
        period_ind = 0
        last_time = 0
        for i in range(len(global_iterations_instants[class_ind])):
            # Activation
            if period_ind + 1 < len(activation_periods):
                while global_iterations_instants[class_ind][i] >= activation_periods[period_ind + 1][0]:
                    period_ind += 1
                    if period_ind >= len(activation_periods) - 1:
                        break
            if (global_iterations_instants[class_ind][i] >= activation_periods[period_ind][0] and
                    global_iterations_instants[class_ind][i] <= activation_periods[period_ind][1]):
                Activations[class_ind][i] = 1

            # Contribution
            if len(triggering_events[class_ind]) > Last_triggering_events_indices[class_ind]:
                z = triggering_events[class_ind][Last_triggering_events_indices[class_ind]]
                if last_time <= z <= global_iterations_instants[class_ind][i]:
                    Candidacies[class_ind][i] = 1
                    Last_triggering_events_indices[class_ind] += 1
            last_time = global_iterations_instants[class_ind][i]

    return Activations, Candidacies


CLASS_NAMES = ("Cheetah", "Ant", "Leg", "Humanoid")


def generate_Xi_matrices(n_octopi, path=None):
    """Load one 4x4 Xi matrix per class for each octopus.

    Rows are global fitness, contributor count, gravity, and 1. Columns
    are initial reward, log mean-reversion rate, log long-term reward,
    and log volatility, as in Assumption 2.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "Model calibration" / "Xi_matrix.json"
    with Path(path).open(encoding="utf-8") as file:
        raw_matrices = json.load(file)
    if set(raw_matrices) != set(CLASS_NAMES):
        raise ValueError(f"Xi matrices must be provided for {', '.join(CLASS_NAMES)}")
    matrices = {name: np.asarray(raw_matrices[name], dtype=float) for name in CLASS_NAMES}
    for name, matrix in matrices.items():
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} Xi matrix must be a finite 4x4 array")
    return [{name: matrix.copy() for name, matrix in matrices.items()} for _ in range(n_octopi)]


def generate_diffusion_parameters(n_octopi):
    """Compatibility alias for notebooks that still use the old name."""
    return generate_Xi_matrices(n_octopi)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)


def get_pareto_indices(mean_rewards: np.ndarray) -> np.ndarray:
    K = len(mean_rewards)
    is_pareto = np.ones(K, dtype=bool)
    for i in range(K):
        for j in range(K):
            if i != j and dominates(mean_rewards[j], mean_rewards[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def length_scalarization(reward_vec: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    x = np.maximum(reward_vec - z, 0.0) / w
    return float(np.min(x))


def hypervolume_2d(front: np.ndarray, ref: np.ndarray = np.array([0.0, 0.0])) -> float:
    if len(front) == 0:
        return 0.0
    pts = np.unique(np.array(front), axis=0)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pts:
        hv += (p[0] - prev_x) * (p[1] - ref[1])
        prev_x = p[0]
    return hv


def layer_init(layer, gain=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ========================== NEURAL NETWORKS (now on CUDA) ==========================
class PolicyNetwork(nn.Module):
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super().__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size), gain=np.sqrt(2))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2))
        self.mu_head = layer_init(nn.Linear(hidden_size, action_shape), gain=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden_size, action_shape), gain=1.0)

    def forward(self, x):
        h = torch.tanh(self.dense1(x))
        h = torch.tanh(self.dense2(h))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        std = torch.exp(log_std.clamp(min=-20, max=2))
        return mu, std


class ValueNetwork(nn.Module):
    def __init__(self, shape_in, hidden_size=64):
        super().__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size), gain=np.sqrt(2))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2))
        self.out = layer_init(nn.Linear(hidden_size, 1), gain=1.0)

    def forward(self, x):
        h = torch.tanh(self.dense1(x))
        h = torch.tanh(self.dense2(h))
        return self.out(h)


class Network(nn.Module):
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super().__init__()
        self.policy = PolicyNetwork(shape_in, action_shape, hidden_size)
        self.value = ValueNetwork(shape_in, hidden_size)

    def forward(self, x):
        return self.policy(x), self.value(x)


# ========================== POLICY (CUDA-AWARE) ==========================
class Policy:
    def __init__(self, model):
        self.model = model.to(device)          # Move model to CUDA

    def act(self, obs, training=False):
        single = len(np.shape(obs)) == 1
        if single:
            obs = np.expand_dims(obs, 0)

        # Convert observation to tensor on correct device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(training):
            (mu, std), v = self.model(x)

        dist = torch.distributions.Normal(mu, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)

        if not training:
            actions = actions.detach().cpu().numpy()
            log_probs = log_probs.detach().cpu().numpy()
            v = v.detach().cpu().numpy().squeeze(-1)

        if single:
            actions = actions[0]
            log_probs = log_probs[0]
            v = v[0]

        if training:
            return {'actions': actions, 'log_probs': log_probs, 'values': v}
        else:
            return {'actions': actions, 'log_probs': log_probs, 'values': v}


# ========================== TRAINING COMPONENTS (CUDA SUPPORT) ==========================
class EnvRunner:
    def __init__(self, env, policy, nsteps, transforms=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.obs, _ = self.env.reset()
        self.total_steps = 0

    def run(self):
        trajectory = defaultdict(list)
        obs = self.obs.copy()

        for _ in range(self.nsteps):
            trajectory['observations'].append(obs)

            act_dict = self.policy.act(obs)
            action = act_dict['actions']

            trajectory['actions'].append(action)
            trajectory['log_probs'].append(act_dict['log_probs'])
            trajectory['values'].append(act_dict['values'])

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            trajectory['rewards'].append(reward)
            trajectory['dones'].append(float(done))

            obs = next_obs.copy() if not done else self.env.reset()[0].copy()
            self.total_steps += 1

        self.obs = obs

        # Convert to numpy arrays (remains on CPU for sampling efficiency)
        for k in ['observations', 'actions', 'log_probs', 'values', 'rewards', 'dones']:
            trajectory[k] = np.asarray(trajectory[k], dtype=np.float32)

        for t in self.transforms:
            t(trajectory)

        return trajectory


class GAE:
    def __init__(self, policy, gamma=0.99, lam=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lam = lam

    def __call__(self, trajectory):
        # Bootstrap value from the last observation
        last_obs = np.expand_dims(trajectory['observations'][-1], 0)
        last_obs_tensor = torch.as_tensor(last_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, v_last = self.policy.model(last_obs_tensor)
        v_last = float(v_last.squeeze().cpu())

        rews = trajectory['rewards']
        dones = trajectory['dones']
        vals = trajectory['values']

        advs = np.zeros(len(rews), dtype=np.float32)
        vtargs = np.zeros(len(rews), dtype=np.float32)

        gae = 0.0
        for t in reversed(range(len(rews))):
            if t == len(rews) - 1:
                delta = rews[t] + self.gamma * v_last * (1 - dones[t]) - vals[t]
            else:
                delta = rews[t] + self.gamma * vals[t + 1] * (1 - dones[t]) - vals[t]

            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs[t] = gae
            vtargs[t] = gae + vals[t]

        trajectory['advantages'] = advs
        trajectory['value_targets'] = vtargs


class NormalizeAdvantages:
    def __call__(self, trajectory):
        adv = trajectory['advantages']
        trajectory['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)


class TrajectorySampler:
    def __init__(self, runner, num_epochs=10, minibatch_size=64, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.transforms = transforms or []
        self.current_traj = None
        self.epoch = 0

    def get_next(self):
        if self.current_traj is None or self.epoch >= self.num_epochs:
            self.current_traj = self.runner.run()
            self.epoch = 0

        traj = self.current_traj
        n = len(traj['rewards'])

        idx = np.random.randint(0, n, size=self.minibatch_size)

        batch = {
            'observations': traj['observations'][idx],
            'actions': traj['actions'][idx],
            'log_probs': traj['log_probs'][idx],
            'values': traj['values'][idx],
            'advantages': traj['advantages'][idx],
            'value_targets': traj['value_targets'][idx],
        }

        self.epoch += 1
        return batch


class PPO:
    def __init__(self, policy, optimizer, cliprange=0.2, vf_coef=0.5, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def compute_loss(self, batch):
        # Move batch to CUDA
        obs = torch.as_tensor(batch['observations'], dtype=torch.float32, device=device)
        old_logp = torch.as_tensor(batch['log_probs'], dtype=torch.float32, device=device)
        adv = torch.as_tensor(batch['advantages'], dtype=torch.float32, device=device)
        vtarg = torch.as_tensor(batch['value_targets'], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.float32, device=device)

        (mu, std), vpred = self.policy.model(obs)
        dist = torch.distributions.Normal(mu, std)
        new_logp = dist.log_prob(actions).sum(-1)

        ratio = torch.exp(new_logp - old_logp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * adv
        policy_loss = -torch.minimum(surr1, surr2).mean()

        value_loss = 0.5 * (vpred.squeeze(-1) - vtarg).pow(2).mean()

        return policy_loss + self.vf_coef * value_loss

    def step(self, batch):
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), gnorm.item()


# ========================== SETUP FUNCTION ==========================
def local_policy_setup(env, model):
    policy = Policy(model)                                   # Model moved to CUDA inside Policy
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    
    runner = EnvRunner(env, policy, nsteps=2048,
                       transforms=[GAE(policy, gamma=0.99, lam=0.95)])
    
    sampler = TrajectorySampler(runner, num_epochs=10, minibatch_size=64,
                                transforms=[NormalizeAdvantages()])
    
    ppo = PPO(policy, optimizer)
    
    return policy, optimizer, runner, sampler, ppo


def federated_average(models, numbers_of_episodes):
    if len(models) == 1:
        return copy.deepcopy(models[0])
    weights = numbers_of_episodes
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("All agents have zero training steps — cannot average.")
    weight_factors = [w / total_weight for w in weights]
    avg_model = copy.deepcopy(models[0])
    for param in avg_model.parameters():
        param.data.zero_()
    for model, factor in zip(models, weight_factors):
        for avg_param, local_param in zip(avg_model.parameters(), model.parameters()):
            avg_param.data += factor * local_param.data
    return avg_model


def get_model_size_in_kb(model):
    if model is None:
        raise ValueError("Model cannot be None.")
    total_size_bytes = 0
    # Sum the size of all parameters
    for param in model.parameters():
        total_size_bytes += param.numel() * param.element_size()
    # Also include buffers (e.g., running statistics in BatchNorm layers)
    for buffer in model.buffers():
        total_size_bytes += buffer.numel() * buffer.element_size()
    # Convert bytes to kilobytes (1 KB = 1024 bytes)
    total_size_kb = total_size_bytes / 1024.0
    return total_size_kb
