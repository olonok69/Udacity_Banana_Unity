
from src.Networks import *
from src.buffers import *
from typing import Dict, List, Tuple
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size replay from Target Networks
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters (when we apply soft update)
LR = 5e-4               # learning rate for Optimizer. Usually, Adam
UPDATE_EVERY = 4        # how often to update the network(only for soft update) for hard update fix
                        # value of 200 and we don’t apply TAU

# if GPU is available move calculations to it
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

class DQN_Agent():
    """
    DQN Agent
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 soft = False,
                 target_update = 200
 ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            soft (bool): use of soft or hard update of target Network. Default false
            target_update (int) : default number of steps to update target network

        """

        self.state_size = state_size # observation or state space size
        self.action_size = action_size # action space size
        self.seed = random.seed(seed)
        self.soft = soft # use soft update. each 4 episodes and apply TAU. # default False then apply hard update

        # Q-Network local and target
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        self.memory = ReplayBuffer(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses= []
        self.all_rewards = []
        self.target_update = target_update
        self.update_cnt = 0

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and self.soft == True: # sampling from Memory
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
        else:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            # explotation phase
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploration phase
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            state = autograd.Variable(torch.FloatTensor(np.float32(states))).to(device)
            next_state = autograd.Variable(torch.FloatTensor(np.float32(next_states))).to(device)
            action = autograd.Variable(torch.LongTensor(actions)).to(device)
            reward = autograd.Variable(torch.FloatTensor(rewards)).to(device)
            done = autograd.Variable(torch.FloatTensor(dones)).to(device)

        # Predict values local DQN and target DQN
        q_values = self.qnetwork_local(state)
        next_q_values = self.qnetwork_target(next_state)

        # transform the dimension of both to single values
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # as target value is a matrix we use max to get the max value of first row
        next_q_value = next_q_values.max(1)[0]

        # calculate expected rewards. If not done consider next_q_value from target network, if done just return
        # reward for that final state
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        #  loss calculation
        loss = F.smooth_l1_loss(q_value, autograd.Variable(expected_q_value.data))
        # record loss for plotting
        self.losses.append(loss.item())

        # We first apply zero_grad to the optimizer to zero out any gradients as a reset. We then push the loss
        # backward, and finally perform one step on the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.soft == False:
            self.update_cnt += 1
            # if hard update is needed
            if self.update_cnt % self.target_update == 0:
                self.target_hard_update()
        else:
            #------------------- update target network ------------------- #
            self.target_soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def target_hard_update(self):
        """
        Hard update: target <- local parameters
        """
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())


    def target_soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DQN_N_steps_Agent():
    """
    DQN N_Steps Learning Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 # N-step Learning
                 n_step: int = 3,
                 train = False,
                 target_update = 200
 ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        #self.memory = ReplayBuffer(BUFFER_SIZE)
        self.n_step = n_step
        # Replay memory N Steps
        self.memory_n = N_Steps_ReplayBuffer(
            self.state_size, BUFFER_SIZE, BATCH_SIZE, n_step=n_step, gamma=GAMMA
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses= []
        self.all_rewards = []
        self.target_update = target_update
        self.update_cnt = 0
        # transition to store in memory
        self.transition = list()
        self.train = train

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory only during training
        if self.train:
            self.transition += [reward, next_state, done]
            self.memory_n.store(*self.transition)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:  # sampling from Memory
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory_n) > BATCH_SIZE:
                    experiences = self.memory_n.sample_batch()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                # use local net to get next action
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            # select next action greedy
            # we need to return the selected action and state from device to CPU
            selected_action = np.argmax(action_values.detach().cpu().data.numpy())
            state = state.detach().cpu().numpy()
            if self.train: # only during training
                self.transition = [state, selected_action]
            return selected_action
        else: # if no greedy use random action from environment
            selected_action = random.choice(np.arange(self.action_size))
            if self.train:
                self.transition = [state, selected_action]
            return selected_action

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """
        self.train = True
        indices = experiences["indices"]
        loss = self._compute_dqn_loss(experiences, gamma)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance.

        samples = self.memory_n.sample_batch_from_idxs(indices)
        gamma = gamma ** self.n_step
        n_loss = self._compute_dqn_loss(samples, gamma)
        loss += n_loss

        # record loss
        self.losses.append(loss.item())
        # backward gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        # if hard update is needed
        if self.update_cnt % self.target_update == 0:
            self._target_hard_update()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def _compute_dqn_loss(
            self,
            samples: Dict[str, np.ndarray],
            gamma: float
    ) -> torch.Tensor:
        """Return dqn loss."""
        # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.qnetwork_local(state).gather(1, action)
        next_q_value = self.qnetwork_target(next_state).max(
            dim=1, keepdim=True
        )[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DDQN_Agent():
    """
    Dueling DQN Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 prioritary_buffer=False,
                 noisy=False,
                 soft=False,
                 target_update=200,
 ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.prioritary_buffer = prioritary_buffer # use prioritary Buffer
        self.noisy = noisy # use Noisy layer
        self.soft = soft  # use soft update. each 4 episodes and apply TAU. # default False then apply hard update
        # Q-Network
        if self.noisy:
            # with Noisy Layer--> remove epsilon greedy
            self.qnetwork_local = Noisy_DuelingNetwork(state_size, action_size).to(device)
            self.qnetwork_target = Noisy_DuelingNetwork(state_size, action_size).to(device)
        else:
            self.qnetwork_local = DDQN(state_size, action_size).to(device)
            self.qnetwork_target = DDQN(state_size, action_size).to(device)
        # initialize target Network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory
        if prioritary_buffer:
            self.memory2 = PrioritizedReplayBuffer(BUFFER_SIZE)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses= []
        self.all_rewards = []
        self.target_update = target_update
        self.update_cnt = 0

    def local_prediction(self, state):
        """
        Predict Q values for given state using local Q network
        :param  state: {array-like} -- Dimension of state space
        returns:[array] -- Predicted Q values for each action in state
        """
        pred = self.qnetwork_local(
            Variable(torch.FloatTensor(state)).to(device)
        )
        pred = pred.data
        return pred

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        if self.prioritary_buffer:
            # Get the timporal difference (TD) error for prioritized replay
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                # Get old Q value. Not that if continous we need to account for batch dimension
                old_q = self.local_prediction(state)[action]

                # Get the new Q value.
                new_q = reward
                if not done:
                    new_q += GAMMA * torch.max(
                        self.qnetwork_target(
                            Variable(torch.FloatTensor(next_state)).to(device)
                        ).data
                    )

                td_error = abs(old_q - new_q)
            self.qnetwork_local.train()
            self.qnetwork_target.train()
            # Save experience in replay memory
            self.memory2.add(td_error.item(), (state, action, reward, next_state, done))
            # Learn every UPDATE_FREQUENCY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0 and self.soft == True: # IF SOFT uPDATE

                # If enough samples are available in memory, get random subset and learn
                if len(self.memory2) > BATCH_SIZE:
                    experiences, idxs, is_weight = self.memory2.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA, idxs, is_weight)
            else:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory2) > BATCH_SIZE:
                    experiences, idxs, is_weight = self.memory2.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA, idxs, is_weight)

        else:
            # Save experience in replay memory
            self.memory.push(state, action, reward, next_state, done)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0 and self.soft==True: # sampling from Memory
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA, 0)
            else:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA, 0)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """
        if self.noisy: # iF NOISY LAYER. we dont use epsilon-greedy algo
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # Choose action values according to local model
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Epsilon-greedy action selection
            if random.random() > eps:
                # explotation
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local(state)
                self.qnetwork_local.train()

                return np.argmax(action_values.cpu().data.numpy())
            else:
                #exploration
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, idxs, is_weight=0):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences
        # Convertions
        states = Variable(torch.Tensor(states)).float().to(device)
        next_states = Variable(torch.Tensor(next_states)).float().to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        is_weight = torch.FloatTensor(is_weight).unsqueeze(1).to(device)

        if self.prioritary_buffer:
            # Dueling Network with Priority buffer
            q_local_argmax = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_argmax).detach()

            # Get Q values for chosen action
            predictions = self.qnetwork_local(states).gather(1, actions)
            # Calculate TD targets
            targets = (rewards + (GAMMA * q_targets_next * (1 - dones)))
            # Update priorities
            errors = torch.abs(predictions - targets).data.cpu().numpy()
            for i in range(len(errors)):
                self.memory2.update(idxs[i], errors[i])

            # Get the loss, using importance sampling weights
            loss = (is_weight * nn.MSELoss(reduction='none')(predictions, targets)).mean()

            # record loss
            self.losses.append(loss.item())

            # Run optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.noisy:
                # NoisyNet: reset noise
                self.qnetwork_local.reset_noise()
                self.qnetwork_target.reset_noise()

            if self.soft == False:
                self.update_cnt += 1
                # if hard update is needed
                if self.update_cnt % self.target_update == 0:
                    self._target_hard_update()
            else:
                # update target network
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        else:
            # Double DQN take the max in between the prediction calculated by the target network and the prediction
            # of the local network using the next_state
            Q_targets_next = self.qnetwork_target(next_states).gather(
                1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()

            # Compute Q targets for current states. same as DQN
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss with smooth for gradient
            # clipping
            loss = F.smooth_l1_loss(Q_expected, Q_targets)
            # record loss
            self.losses.append(loss.item())

            # Minimize the loss and backpropagate it
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # NoisyNet: reset noise
            if self.noisy:

                self.qnetwork_local.reset_noise()
                self.qnetwork_target.reset_noise()

            if self.soft == False:
                self.update_cnt += 1
                # if hard update is needed
                if self.update_cnt % self.target_update == 0:
                    self._target_hard_update()
            else:
                # update target network
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _target_hard_update(self):
        """
        Hard update: target <- local parameters
        """
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Cat_DQN_Agent():
    """
    DQN Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 train=False,
                 target_update= 200,
                 # Categorical DQN parameters
                 v_min: float = 0.0,
                 v_max: float = 200.0,
                 atom_size: int = 51
 ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(device)

        # Q-Network
        self.qnetwork_local = Categorical_DQN(state_size, action_size, atom_size, self.support).to(device)
        self.qnetwork_target = Categorical_DQN(state_size, action_size, atom_size, self.support).to(device)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        self.memory = NPReplayBuffer(self.state_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses= []
        self.all_rewards = []
        # transition to store in memory
        self.transition = list()
        self.train = train
        self.target_update = target_update
        self.update_cnt = 0

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        if self.train:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0: # sampling from Memory
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample_batch()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            # select next action greedy
            selected_action = np.argmax(action_values.detach().cpu().data.numpy())
            state = state.detach().cpu().numpy()
            if self.train:
                self.transition = [state, selected_action]
            return selected_action
        else:
            selected_action = random.choice(np.arange(self.action_size))
            if self.train:
                self.transition = [state, selected_action]
            return selected_action

    def learn(self, samples, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """
        self.train = True

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.qnetwork_target(next_state).argmax(1)
            next_dist = self.qnetwork_target.dist(next_state)
            next_dist = next_dist[range(BATCH_SIZE), next_action]

            t_z = reward + (1 - done) * GAMMA* self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (BATCH_SIZE - 1) * self.atom_size, BATCH_SIZE
                ).long()
                    .unsqueeze(1)
                    .expand(BATCH_SIZE, self.atom_size)
                    .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.qnetwork_local.dist(state)
        log_p = torch.log(dist[range(BATCH_SIZE), action])

        loss = -(proj_dist * log_p).sum(1).mean()
        # record loss
        self.losses.append((loss.item()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_cnt += 1
        # if hard update is needed
        if self.update_cnt % self.target_update == 0:
            self._target_hard_update()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DQN_Rainbow_Agent():
    """
    DQN Rainbow Learning Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 # N-step Learning
                 n_step: int = 3,
                 # PER parameters
                 alpha: float = 0.2,
                 beta: float = 0.6,
                 prior_eps: float = 1e-6,
                 # Categorical parameters
                 v_min: float = 0.0,
                 v_max: float = 200.0,
                 atom_size: int = 51,
                 num_frames=2000,
                 train = False,
                 target_update = 100
 ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_frames = num_frames

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = N_Steps_PrioritizedReplayBuffer(
            self.state_size, BUFFER_SIZE, BATCH_SIZE, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = N_Steps_ReplayBuffer(
                self.state_size, BUFFER_SIZE, BATCH_SIZE, n_step=n_step, gamma=GAMMA
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(device)

        # Q-Network
        self.qnetwork_local = Rainbow_DQN(state_size, action_size, self.atom_size, self.support).to(device)
        self.qnetwork_target = Rainbow_DQN(state_size, action_size, self.atom_size, self.support).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        #self.memory = ReplayBuffer(BUFFER_SIZE)
        self.n_step = n_step
        # Replay memory N Steps
        self.memory_n = N_Steps_ReplayBuffer(
            self.state_size, BUFFER_SIZE, BATCH_SIZE, n_step=n_step, gamma=GAMMA
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses= []
        self.all_rewards = []
        self.target_update = target_update
        self.update_cnt = 0
        # transition to store in memory
        self.transition = list()
        self.train = train

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        if self.train:
            # PER: increase beta
            fraction = min(self.update_cnt / self.num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            self.transition += [reward, next_state, done]
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:  # sampling from Memory
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory_n) > BATCH_SIZE:
                    experiences = self.memory.sample_batch(self.beta)
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        # NoisyNet: no epsilon greedy action selection

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # select next action greedy
        selected_action = np.argmax(action_values.detach().cpu().data.numpy())
        state = state.detach().cpu().numpy()
        if self.train:
            self.transition = [state, selected_action]
        return selected_action


    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """
        self.train = True
        # PER needs beta to calculate weights
        weights = torch.FloatTensor(
            experiences["weights"].reshape(-1, 1)
        ).to(device)
        indices = experiences["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(experiences, gamma)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = GAMMA ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        # record loss
        self.losses.append(loss.item())
        # backward gradients
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        # if hard update is needed
        self.update_cnt += 1

        if self.update_cnt % self.target_update == 0:
            self._target_hard_update()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
          # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.qnetwork_local(next_state).argmax(1)
            next_dist = self.qnetwork_target.dist(next_state)
            next_dist = next_dist[range(BATCH_SIZE), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (BATCH_SIZE - 1) * self.atom_size, BATCH_SIZE
                ).long()
                    .unsqueeze(1)
                    .expand(BATCH_SIZE, self.atom_size)
                    .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.qnetwork_local.dist(state)
        log_p = torch.log(dist[range(BATCH_SIZE), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def dqn_runner(env,
        brain_name: str,
        agent,
        algo: int,
        n_episodes : int,
        max_t : int,
        eps_start : float,
        eps_end : float,
        eps_decay : float) -> Tuple[list, deque, int, List]:
    """
    Run a game and capture statistics

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    losses= []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    epi = 0 # number of episodes
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(int)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:

                break
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score
        losses.append(np.mean(agent.losses))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 16.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(
                                                                                             scores_window)))
            epi = i_episode
            torch.save(agent.qnetwork_local.state_dict(), f'models\checkpoint_{algo}.pth')
            break
        epi = i_episode
    return scores, scores_window, epi, losses

def all_dqn_runner(env,
        brain_name: str,
        agent,
        algo: int,
        n_episodes : int,
        max_t : int,
        eps_start : float,
        eps_end : float,
        eps_decay : float) -> Tuple[list, deque, int, List]:

    """
    Run a game and capture statistics

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    losses= []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    epi = 0 # number of episodes

    max_score = 13.0
    win = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(int)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:

                break
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score
        losses.append(np.mean(agent.losses))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13 and win == False:
            # if we reach 13 , we win the game
            epi = i_episode
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(
                                                                                             scores_window)))
            win = True
        if np.mean(scores_window) >= max_score :
            # save the best model
            torch.save(agent.qnetwork_local.state_dict(), f'models\checkpoint_{algo}.pth')

            max_score = np.mean(scores_window)

        #epi = i_episode
        # if np.mean(scores)> 13:
        #     torch.save(agent.qnetwork_local.state_dict(), f'models\checkpoint_{algo}.pth')
    return scores, scores_window, epi, losses