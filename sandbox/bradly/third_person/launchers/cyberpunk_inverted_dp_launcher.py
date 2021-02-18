import sys
sys.path.append('/home/asus/Workspace/Third_Person/')
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.policy.random_policy import RandomPolicy
from sandbox.bradly.third_person.algos.cyberpunk_trainer import CyberPunkTrainer
from sandbox.bradly.third_person.policy.expert_inverted_dp import load_expert
from sandbox.bradly.third_person.envs.inverted_dp import InvertedPendulumEnv
from sandbox.bradly.third_person.envs.inverted_dp_two import InvertedPendulumTwoEnv
from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionVelocityDiscriminator

import seaborn as sns
import tensorflow as tf
import os
from pandas import DataFrame
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import copy

def seed_tensorflow(seed=42):
    # tf.reset_default_graph()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

exp_idx=2
seed =2000
im_size = 50
im_channels = 3
n_trajs_cost=32
n_trajs_policy=32
iterations=5000
train =True

seed_tensorflow(seed)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(allow_growth=True)

expert_env = TfEnv(normalize(InvertedPendulumEnv(size=im_size)))
novice_env = TfEnv(normalize(InvertedPendulumTwoEnv(size=im_size), normalize_obs=True))
expert_fail_pol = RandomPolicy(expert_env.spec)

policy = GaussianMLPPolicy(
    name="novice_policy",
    env_spec=novice_env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=expert_env.spec)

algo = TRPO(
    env=novice_env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=50,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

)
  
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    algo.n_itr = 0
    algo.start_itr = 0
    algo.train(sess=sess)
 
    # im_size = 50
    # im_channels = 3

    dim_input = [im_size, im_size, im_channels]
    disc = DomainConfusionVelocityDiscriminator(input_dim=dim_input, output_dim_class=2, output_dim_dom=2,
                                                tf_sess=sess)

    expert_policy = load_expert()

    #from rllab.sampler.utils import rollout
    #while True:
    #        t = rollout(env=expert_env, agent=expert_policy, max_path_length=50, animated=True)
    #        print(t['observations'].shape)

    algo.n_itr = 40
    trainer = CyberPunkTrainer(disc=disc, novice_policy_env=novice_env, expert_fail_pol=expert_fail_pol,
                               expert_env=expert_env, novice_policy=policy,
                               novice_policy_opt_algo=algo, expert_success_pol=expert_policy,
                               im_width=im_size, im_height=im_size, im_channels=im_channels,
                               tf_sess=sess, horizon=50,seed =exp_idx, n_trajs_cost= n_trajs_cost)    
    # iterations = 10
    tf.get_variable_scope().reuse_variables()
    
    if train:
        best_rew=-float("inf")
        res_np = np.empty((0,2))
        bestpol = copy.deepcopy(policy)
        for iter_step in range(0, iterations):
            # train
            train_path=trainer.take_iteration(n_trajs_cost=n_trajs_cost, n_trajs_policy=n_trajs_policy)
            trainer.log_and_finish()
            # eval
            pol=trainer.novice_policy
            ckpt_path=trainer.collect_trajs_for_policy(n_trajs=10, pol=pol, env=trainer.novice_policy_env,animated=True)
            ckpt_rew = [sum(path['true_rewards']) for path in ckpt_path]
            # save ckpt
            ckpt_rew_mean =sum(ckpt_rew)/len(ckpt_rew)
            if ckpt_rew_mean> best_rew:
                best_rew = ckpt_rew_mean
                bestpol = copy.deepcopy(pol)
                with open(f'stu_cartpole_{seed}_v{exp_idx}.pickle', 'wb') as handle:
                    pickle.dump(bestpol, handle)
            print("current best rew: ",best_rew)
            # save data
            iter_c = [iter_step]*len(ckpt_rew)
            res = np.array(list(zip(iter_c,ckpt_rew)))
            res_np = np.append(res_np,res,axis=0)
            with open(f'train_cartpole_{seed}_v{exp_idx}.npy', 'wb') as f:
                np.save(f, res_np)
            # plot figure
            plt.figure()
            sns.set_theme()
            df = DataFrame(res_np,columns=['epoch','reward'])
            plot =sns.lineplot(data=df,x='epoch',y='reward',ci="sd")
            locator = ticker.MultipleLocator(2)
            plot.xaxis.set_major_locator(locator)
            plot.set_title('Cartpole Third Person')
            fig=plot.get_figure()
            fig.savefig(f'res_cartpole_{seed}_v{exp_idx}.png')
    else:
        # test
        res_policy = load_expert(f=f'stu_cartpole_{seed}_v{exp_idx}.pickle')
        ckpt_path=trainer.collect_trajs_for_policy(n_trajs=50, pol=res_policy, env=trainer.novice_policy_env)
        ckpt_rew = [sum(path['true_rewards']) for path in ckpt_path]
        ckpt_rew =sum(ckpt_rew)/len(ckpt_rew)
        print("best ckpt test reward: ",ckpt_rew)