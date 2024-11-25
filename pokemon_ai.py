import numpy as np
from pyboy import PyBoy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Define button constants
BUTTONS = {
    'UP': 0x40,
    'DOWN': 0x80,
    'LEFT': 0x20,
    'RIGHT': 0x10,
    'A': 0x01,
    'B': 0x02,
    'START': 0x08,
    'SELECT': 0x04
}

# Memory addresses
MEMORY_ADDRS = {
    'PLAYER_X': 0xD362,
    'PLAYER_Y': 0xD361,
    'MAP_ID': 0xD35E,
    'BATTLE_TYPE': 0xD057,
    'PARTY_COUNT': 0xD163
}

class PokemonAI:
    def __init__(self, rom_path="/Users/danieltomaro/Documents/Projects/Pokemon/ROMs/Pokemon Red.gb", 
                 memory_size=10000, batch_size=32):
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found at {rom_path}")
            
        print(f"Initializing PyBoy with ROM at {rom_path}")
        
        self.rom_path = rom_path
        self.pyboy = self._init_pyboy()
        
        # Create directory for saving models and logs
        self.save_dir = "pokemon_ai_saves"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_steps = []
        self.exploration_rates = []
        
        # Define possible actions
        self.actions = list(BUTTONS.keys())
        
        # Initialize state tracking
        self.last_x = None
        self.last_y = None
        self.last_map = None
        self.steps_since_progress = 0
        self.battle_state = False
        
        # Initialize frame buffer for state history
        self.frame_buffer = deque(maxlen=4)
        
        # Get initial screen to determine dimensions
        initial_screen = self.pyboy.screen.ndarray
        self.screen_height, self.screen_width = initial_screen.shape[:2]
        
        # Calculate conv output size
        h = self.screen_height
        w = self.screen_width
        h = (h - 8) // 4 + 1  # First conv
        w = (w - 8) // 4 + 1
        h = (h - 4) // 2 + 1  # Second conv
        w = (w - 4) // 2 + 1
        h = (h - 3) + 1  # Third conv
        w = (w - 3) + 1
        self.conv_output_size = 64 * h * w
        
        # DQN parameters
        print("Initializing DQN parameters...")
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=memory_size)
        
        # Initialize neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())

    def _init_pyboy(self):
        pyboy = PyBoy(
            self.rom_path,
            window="SDL2"
        )
        pyboy.set_emulation_speed(0)
        return pyboy

    def _build_model(self):
        return nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, len(self.actions))
        )

    def get_state(self):
        try:
            screen = self.pyboy.screen.ndarray
            if len(screen.shape) == 3:
                screen = screen.mean(axis=2)
            screen = screen.astype(np.float32) / 255.0
            
            self.frame_buffer.append(screen)
            while len(self.frame_buffer) < 4:
                self.frame_buffer.append(screen)
            
            state = np.stack(list(self.frame_buffer))
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error getting state: {e}")
            raise

    def perform_action(self, action):
        try:
            button_value = BUTTONS[action]
            # Press button
            self.pyboy.send_input(button_value)
            self.pyboy.tick()
            # Release button
            self.pyboy.send_input(0)
            self.pyboy.tick()
        except Exception as e:
            print(f"Error performing action {action}: {e}")

    def calculate_reward(self):
        try:
            reward = 0
            
            # Get current state from memory
            player_x = self.pyboy.memory[MEMORY_ADDRS['PLAYER_X']]
            player_y = self.pyboy.memory[MEMORY_ADDRS['PLAYER_Y']]
            current_map = self.pyboy.memory[MEMORY_ADDRS['MAP_ID']]
            
            # Initialize if first step
            if self.last_x is None:
                self.last_x = player_x
                self.last_y = player_y
                self.last_map = current_map
                return 0
            
            # Check if in battle
            in_battle = self.detect_battle()
            if in_battle != self.battle_state:
                if in_battle:
                    reward += 5  # Reward for entering battle
                self.battle_state = in_battle
            
            # Reward movement
            if player_x != self.last_x or player_y != self.last_y:
                reward += 1
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1
                if self.steps_since_progress > 10:
                    reward -= 0.5
            
            # Reward map changes
            if current_map != self.last_map:
                reward += 20
                print(f"Entered new map! ID: {current_map}")
            
            # Check party size
            party_count = self.pyboy.memory[MEMORY_ADDRS['PARTY_COUNT']]
            if party_count > 1:
                reward += 50
            
            self.last_x = player_x
            self.last_y = player_y
            self.last_map = current_map
            
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0

    def detect_battle(self):
        try:
            return self.pyboy.memory[MEMORY_ADDRS['BATTLE_TYPE']] != 0
        except Exception as e:
            print(f"Error detecting battle: {e}")
            return False

    def reset_game(self):
        try:
            # Stop any existing PyBoy instance
            if hasattr(self, 'pyboy'):
                self.pyboy.stop()

            # Reinitialize PyBoy
            self.pyboy = self._init_pyboy()
            self.frame_buffer.clear()
            
            # Initialize state tracking
            self.last_x = None
            self.last_y = None
            self.last_map = None
            self.steps_since_progress = 0
            self.battle_state = False

            # Spam "A" button to bypass the main menu
            print("Spamming 'A' button to pass the main menu...")
            for _ in range(300):  # Adjust the range to ensure sufficient duration
                self.perform_action('A')
                self.pyboy.tick()
            
            print("Game initialization complete.")
            return self.get_state()
        except Exception as e:
            print(f"Error resetting game: {e}")
            raise

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        with torch.no_grad():
            q_values = self.model(state)
            return self.actions[torch.argmax(q_values).item()]

    def remember(self, state, action, reward, next_state, done):
            """
            Store experience in replay memory.
            
            Args:
                state: Current state (tensor)
                action: Action taken (string)
                reward: Reward received (float)
                next_state: Next state (tensor)
                done: Whether episode ended (bool)
            """
            try:
                self.memory.append((state, action, reward, next_state, done))
                if len(self.memory) % 1000 == 0:  # Log every 1000 memories stored
                    print(f"Replay buffer size: {len(self.memory)}/{self.memory_size}")
            except Exception as e:
                print(f"Error storing experience in memory: {e}")
                print(f"State shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Next state shape: {next_state.shape if hasattr(next_state, 'shape') else 'unknown'}")
                print(f"Done: {done}")
                import traceback
                traceback.print_exc()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        try:
            minibatch = random.sample(self.memory, self.batch_size)
            
            states = torch.cat([m[0] for m in minibatch])
            actions = torch.tensor([self.actions.index(m[1]) for m in minibatch], device=self.device)
            rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32, device=self.device)
            next_states = torch.cat([m[3] for m in minibatch])
            dones = torch.tensor([m[4] for m in minibatch], dtype=torch.float32, device=self.device)
            
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_model(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        except Exception as e:
            print(f"Error in replay: {e}")
            import traceback
            traceback.print_exc()

    def train(self, steps_per_save=10000):
        """
        Continuously train the AI without resetting for episodes.
        The process runs indefinitely until interrupted (e.g., via Ctrl+C).
        
        Args:
            steps_per_save: Number of steps to run before saving progress.
        """
        print("Starting continuous training...")
        print("Press Ctrl+C to stop the training process.")
        
        try:
            step_count = 0
            total_reward = 0
            state = self.get_state()  # Get the initial state of the game
            
            while True:  # Infinite loop for continuous training
                step_count += 1

                # Choose an action and perform it
                action = self.act(state)
                self.perform_action(action)

                # Observe new state and calculate reward
                next_state = self.get_state()
                reward = self.calculate_reward()
                total_reward += reward
                done = self.steps_since_progress > 100  # Define a condition to reset progress if needed

                # Store experience in memory and learn
                self.remember(state, action, reward, next_state, done)
                self.replay()

                # Update current state
                state = next_state

                # Periodically save progress
                if step_count % steps_per_save == 0:
                    print(f"Steps: {step_count}, Total Reward: {total_reward}")
                    self.save_progress(step_count)
                    self.plot_metrics()

                # Optionally, handle reset conditions
                if done:
                    print("Stuck detected. Attempting recovery by spamming 'A'...")
                    for _ in range(50):  # Spam 'A' to try recovering from the stuck state
                        self.perform_action('A')
                        self.pyboy.tick()
                    state = self.get_state()  # Re-fetch the current state after recovery
                    total_reward = 0
                    self.steps_since_progress = 0  # Reset progress counter

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving progress...")
            self.save_progress(step_count)
            print("Progress saved. Training stopped.")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_progress(self, episode):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"model_episode_{episode}_{timestamp}.pth")
            
            save_data = {
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_steps': self.episode_steps,
                'exploration_rates': self.exploration_rates,
                'conv_output_size': self.conv_output_size,
                'screen_dimensions': (self.screen_height, self.screen_width)
            }
            
            torch.save(save_data, save_path)
            print(f"Progress saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving progress: {e}")
            import traceback
            traceback.print_exc()

    def load_progress(self, filepath):
        try:
            print(f"Loading model from {filepath}")
            checkpoint = torch.load(filepath)
            
            # Verify compatible dimensions
            if (checkpoint['screen_dimensions'] != (self.screen_height, self.screen_width) or
                checkpoint['conv_output_size'] != self.conv_output_size):
                raise ValueError("Saved model dimensions don't match current settings")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_steps = checkpoint['episode_steps']
            self.exploration_rates = checkpoint['exploration_rates']
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def plot_metrics(self):
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot rewards
            plt.subplot(1, 3, 1)
            plt.plot(self.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            
            # Plot steps
            plt.subplot(1, 3, 2)
            plt.plot(self.episode_steps)
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            # Plot exploration rate
            plt.subplot(1, 3, 3)
            plt.plot(self.exploration_rates)
            plt.title('Exploration Rate')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            
            plt.tight_layout()
            plot_path = os.path.join(self.save_dir, 'training_metrics.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Training metrics plotted to {plot_path}")
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        print("Creating PokemonAI instance...")
        ai = PokemonAI()
        print("Starting training process...")
        ai.train(episodes=100, steps_per_episode=1000, save_interval=10)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'ai' in locals() and hasattr(ai, 'pyboy'):
            ai.pyboy.stop()
        print("Cleanup complete")