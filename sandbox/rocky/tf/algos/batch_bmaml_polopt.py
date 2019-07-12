import matplotlib
matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.bmaml_batch_sampler import BmamlBatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.spaces import Discrete
from rllab.sampler.stateful_pool import singleton_pool
from sandbox.rocky.tf.misc import tensor_utils, svpg_tf_utils

class BatchBMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy_list,
            baseline_list,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            batch_size=100,
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            num_leader_grad_updates=2,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            load_policy=None,
            svpg_alpha=1.0,
            random_seed=1,
            svpg=True,
            evolution=False,
            evol_step=10,
            evol_ratio=0.5,
            evol_epsilon=0.01,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy_list: Policy list
        :param baseline_list: Baseline list
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy_list = policy_list
        self.load_policy=load_policy
        self.baseline_list = baseline_list
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size # number of tasks
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.num_leader_grad_updates = num_leader_grad_updates
        self.n_particles = len(self.policy_list)  # number of particle
        self.svpg_alpha = svpg_alpha
        self.random_seed = random_seed
        self.svpg = svpg
        self.evolution = evolution
        self.evol_step = evol_step
        self.evol_ratio = evol_ratio
        self.evol_epsilon = evol_epsilon
        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BmamlBatchSampler
            else:
                raise ValueError    # num_parallel > 1
                sampler_cls = VectorizedSampler
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', log=True):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log)

    def make_flat(self,a_list):
        """
        implemented in bmaml_npo.py
        """
        raise NotImplementedError

    def kernel(self, particle_tensor):
        """
        implemented in bmaml_npo.py
        """
        raise NotImplementedError

    def compute_svpg(self, total_grads_list, total_params_list):
        num_param = len(total_grads_list[0][0])
        total_svpg_list = []  # meta_batch_size x num_particles x num_param
        for i, grads_list, params_list in zip(range(len(total_grads_list)), total_grads_list, total_params_list):
            f_g_list = self.make_flat(grads_list)
            f_p_list = self.make_flat(params_list)
            if len(self.policy_list) == 1:
                grad = f_g_list[0]
            else:
                kernel_mat,grad_kernel,h=self.kernel(f_p_list)
                if self.svpg:  # SVPG
                    grad=(tf.matmul(kernel_mat,((1/self.svpg_alpha)*f_g_list))-grad_kernel)/len(self.policy_list)
                else:  # VPG
                    grad=f_g_list
            if i == 0:
				# get original shape (2 is for flat version)
                origin_shape=np.zeros(num_param, dtype=object)
                origin_shape2=np.zeros(num_param, dtype=object)
                for param_idx in range(num_param):
                    params_shape=grads_list[0][param_idx].get_shape().as_list()
                    total_len=1
                    for param_shape in params_shape:
                        total_len *= param_shape
                    origin_shape[param_idx]=params_shape
                    origin_shape2[param_idx]=total_len
            # reshape gradient
            if (len(self.policy_list) > 1):
                grad=tf.unstack(grad,axis=0)
            else:
                grad=[grad]
            grad_list2=[]
            for n in range(len(total_grads_list[0])):
                st_idx=0;length=0;
                grad_list2_part = []
                for param_idx in range(num_param):
                    st_idx+=length;length=origin_shape2[param_idx]
                    grad_list2_part.append(tf.reshape(tf.slice(grad[n],[st_idx],[length]),origin_shape[param_idx]))
                grad_list2.append(grad_list2_part)
            total_svpg_list.append(grad_list2)

        total_svpg_list_reshape = []  # num_particles x meta_batch_size x num_param
        for n in range(len(total_grads_list[0])):
            sub_total_svpg_list_reshape = []
            for i in range(len(total_grads_list)):
                sub_total_svpg_list_reshape.append(total_svpg_list[i][n])
            total_svpg_list_reshape.append(sub_total_svpg_list_reshape)
        return total_svpg_list_reshape

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:
            # Code for loading a previous policy
            if self.load_policy is not None:
                import joblib
                load_data = joblib.load(self.load_policy+'0.pkl')
                self.start_itr = int(load_data['itr'])
                self.policy_list[0] = load_data['policy']
                for n in range(1, self.n_particles):
                    self.policy_list[n] = joblib.load(self.load_policy+str(n)+'.pkl')['policy']
            self.init_opt()
            
            # few-shot computation graph construction
            grads_list, params_list = [], [] # num_particles x meta_batch_size x num_param
            for self.policy in self.policy_list:
                g_list, p_list = self.policy.compute_grads(self.meta_batch_size)
                grads_list.append(g_list)
                params_list.append(p_list)

            grads_list_reshape, params_list_reshape = [], []  # meta_batch_size x num_particles x num_param
            for i in range(self.meta_batch_size):
                sub_grads_list_reshape, sub_params_list_reshape = [], []
                for n in range(len(self.policy_list)):
                    sub_grads_list_reshape.append(grads_list[n][i])
                    sub_params_list_reshape.append(params_list[n][i])
                grads_list_reshape.append(sub_grads_list_reshape)
                params_list_reshape.append(sub_params_list_reshape)

            svpg_grads_list = self.compute_svpg(grads_list_reshape, params_list_reshape)
            for n, self.policy in zip(range(len(self.policy_list)), self.policy_list):
                self.policy.compute_updated_dists_first(svpg_grads_list[n], self.meta_batch_size)
            
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            # set random seed
            np.random.seed(self.random_seed)
            seed_list = np.random.randint(self.n_itr, size=self.n_itr)

            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr+1, self.n_itr+1):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):

                    env = self.env
                    while 'sample_goals' not in dir(env):
                        env = env.wrapped_env
                    np.random.seed(seed_list[itr-1])
                    learner_env_goals = env.sample_goals(self.meta_batch_size)

                    for self.policy in self.policy_list:
                        self.policy.switch_to_init_dist()  # Switch to pre-update policy

                    step_particle_rewards = np.zeros(self.n_particles, dtype=float)
                    step_samples_data = np.zeros(self.n_particles, dtype=object)
                    all_samples_data = np.zeros(self.n_particles, dtype=object)
                    for n in range(self.n_particles):
                        all_samples_data[n] = []
                    for step in range(self.num_grad_updates+1):
                        paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix=str(step))
                        for n, self.policy, self.baseline in zip(range(self.n_particles), self.policy_list, self.baseline_list):
                            samples_data, task_rewards = {}, 0.0
                            for key_idx in range(self.meta_batch_size):  # the keys are the tasks
                                key = str(n) + "_" + str(key_idx)
                                samples_data[key_idx], task_reward = self.process_samples(itr, paths[key], prefix=str(step) + '_' + str(n) + '_' + str(key_idx), log=True)
                                task_rewards += task_reward
                            step_particle_rewards[n] = task_rewards / self.meta_batch_size
                            all_samples_data[n].append(samples_data)
                            step_samples_data[n] = samples_data
                        if step < self.num_grad_updates:
                            logger.log("Computing policy updates...")
                            for self.policy in self.policy_list:
                                self.policy.compute_updated_dists(step_samples_data)

                    logger.log("Meta Learning")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    for n, self.policy, self.baseline in zip(range(self.n_particles), self.policy_list, self.baseline_list):
                        self.optimize_policy(itr, all_samples_data, n)
                        logger.log("Saving " + str(n) + "-th particle snapshot...")
                        params = self.get_itr_snapshot(itr, all_samples_data[n][-1])  # , **kwargs)
                        if self.store_paths:
                            params["paths"] = all_samples_data[-1]["paths"]
                        logger.save_itr_params(itr, params, postfix="_"+str(n))
                        logger.log("Saved")
                    logger.dump_tabular(load_policy=self.load_policy, with_prefix=False)

                    # evolution
                    if ((itr) % self.evol_step == 0) and self.evolution:
                        sorted_id = np.argsort(step_particle_rewards)
                        num_of_evolution = int(len(self.policy_list) * self.evol_ratio)
                        deleted_id = sorted_id[:num_of_evolution]
                        sampled_id = sorted_id[num_of_evolution:]
                        for d_id_idx in range(len(deleted_id)):
                            current_id = np.random.choice(sampled_id, 1)[0]
                            current_params = self.policy_list[current_id].get_param_values()
                            current_epsilon = self.evol_epsilon * (np.random.random(current_params.shape) - 0.5)
                            self.policy_list[deleted_id[d_id_idx]].set_param_values(current_params + current_epsilon)
                        

        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
