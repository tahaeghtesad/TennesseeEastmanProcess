from trainer import Trainer

trainer = Trainer(total_training_steps=500 * 1000, env='BRP', exploration=0.0,)
trainer.bootstrap_defender()


# import gym
# import gym_control
#
# env = gym.make('Historitized-v0', env='BRP-v0')
# obs = env.reset()
# for i in range(1000):
#     obs, reward, done, info = env.step(env.action_space.sample())
#     print(f'{i} ~> {obs}\t{reward:.5f}\t{done}')
#
#     if done:
#         obs = env.reset()
