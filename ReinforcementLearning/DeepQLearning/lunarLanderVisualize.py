from simple_dqn_keras import Agent
import numpy as np
import gym

if __name__ == '__main__':
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8,
                n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.0)
    agent.load_model()
    env = gym.make('LunarLander-v2')
    observation = env.reset()
    agent.q_eval.summary()

    for i in range(5):
        score = 0
        for _ in range(500):
            env.render()
            agent.epsilon = 0
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print(score)
                break
        observation = env.reset()
    env.close()