from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_bmaml_polopt import BatchBMAMLPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils, svpg_tf_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np
from collections import OrderedDict

class BMAMLREPTILE(BatchBMAMLPolopt, Serializable):

    def __init__(
            self,
            env,
            policy_list,
            baseline_list,
            optimizer=None,
            optimizer_args=None,
            use_maml=True,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer_list = [FirstOrderOptimizer(**optimizer_args) for n in range(len(policy_list))]
        self.optimizer_list = optimizer_list
        self.opt_info = None
        super(BMAMLREPTILE, self).__init__(env=env, policy_list=policy_list, baseline_list=baseline_list, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, adv_vars = [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
        return obs_vars, action_vars, adv_vars

    def make_flat(self,a_list):
        a_list2=np.zeros((len(a_list),len(a_list[0])),dtype=object);
        for i in range(len(a_list)):
            for j in range(len(a_list[0])):
                a_list2[i,j]=tf.reshape(a_list[i][j],[-1]);
        a_flat_list=[];
        for i in range(len(a_list2)):
            a_flat_list.append(tf.concat(list(a_list2[i]),axis=0));
        return tf.stack(a_flat_list,axis=0);

    def kernel(self, particle_tensor):
        # kernel
        h = -1
        euclidean_dists = svpg_tf_utils.pdist(particle_tensor)
        pairwise_dists = svpg_tf_utils.squareform(euclidean_dists) ** 2
        # kernel trick
        mean_dist = tf.reduce_mean(euclidean_dists) ** 2
        h = tf.sqrt(0.5 * mean_dist / tf.log(self.n_particles + 1.))

        kernel_matrix = tf.exp(-pairwise_dists / h ** 2 / 2)
        kernel_sum = tf.reduce_sum(kernel_matrix, axis=1)
        grad_kernel = tf.add(-tf.matmul(kernel_matrix, particle_tensor),
                             tf.multiply(particle_tensor, tf.expand_dims(kernel_sum, axis=1))) / (h ** 2)
        return kernel_matrix, grad_kernel, h
        """
        h = tf.sqrt(0.5 * svpg_tf_utils.median(pairwise_dists) / tf.log(self.n_particles + 1.))
        kernel_matrix = tf.exp(-pairwise_dists / h ** 2 / 2)
        kernel_sum = tf.reduce_sum(kernel_matrix, axis=1)
        grad_kernel = tf.add(-tf.matmul(kernel_matrix, particle_tensor),tf.multiply(particle_tensor, tf.expand_dims(kernel_sum, axis=1))) / (h ** 2)
        return kernel_matrix, grad_kernel
        """

    @overrides
    def init_opt(self):

        # get old dist for calculating KL
        old_dist_info_vars, old_dist_info_vars_list, dist_list = [], [], []
        for self.policy in self.policy_list:
            is_recurrent = int(self.policy.recurrent)
            assert not is_recurrent # not supported right now.
            dist = self.policy.distribution
            dist_list.append(dist)

            sub_old_dist_info_vars, sub_old_dist_info_vars_list = [], []
            for i in range(self.meta_batch_size):
                sub_old_dist_info_vars.append({
                    k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                    for k, shape in dist.dist_info_specs
                    })
                sub_old_dist_info_vars_list += [sub_old_dist_info_vars[i][k] for k in dist.dist_info_keys]
            old_dist_info_vars_list.append(sub_old_dist_info_vars_list)

        state_info_vars, state_info_vars_list = {}, []

        input_list = []
        new_params_list = []  # meta_batch_size x num_particles
        for i in range(self.meta_batch_size):
            new_params_list.append([self.policy_list[n].all_params for n in range(len(self.policy_list))])
        global_params_list = new_params_list
        param_keys = list(self.policy_list[0].all_params.keys())
        num_param = len(param_keys)

        # h
        self.global_h = 0.0
        for j in range(self.num_leader_grad_updates):
            obs_vars_list, action_vars_list, adv_vars_list = [], [], []
            for n in range(len(self.policy_list)):
                obs_vars, action_vars, adv_vars = self.make_vars(str(j))
                obs_vars_list.append(obs_vars)
                action_vars_list.append(action_vars)
                adv_vars_list.append(adv_vars)

            surr_objs_list = np.zeros(len(self.policy_list), dtype=object)
            for n in range(len(self.policy_list)):
                surr_objs_list[n] = []

            cur_params_list = new_params_list
            new_params_list = []

            # get follower params
            if j == self.num_grad_updates:
                follower_params_list = cur_params_list

            for i in range(self.meta_batch_size):
                grads_list, params_list = [], []
                for n, self.policy in zip(range(len(self.policy_list)), self.policy_list):
                    dist_info_vars, _ = self.policy.dist_info_sym(obs_vars_list[n][i], state_info_vars, all_params=cur_params_list[i][n])
                    logli = dist_list[n].log_likelihood_sym(action_vars_list[n][i], dist_info_vars)
                    surr_objs = - tf.reduce_mean(logli * adv_vars_list[n][i])
                    surr_objs_list[n].append(surr_objs)
                    grads_list.append(tf.gradients(surr_objs,[cur_params_list[i][n][key] for key in param_keys]))
                    params_list.append([cur_params_list[i][n][key] for key in param_keys])
                all_params_flat = self.make_flat(params_list)
                gradients_flat = self.make_flat(grads_list)
                if(self.n_particles==1):
                    grad=gradients_flat[0]
                else:
                    kernel_mat,grad_kernel,h=self.kernel(all_params_flat)
                    if self.svpg:  # SVPG
                        grad=(tf.matmul(kernel_mat,((1/self.svpg_alpha)*gradients_flat))-grad_kernel)/len(self.policy_list)
                    else:  # VPG
                        grad=gradients_flat
                if (j==0) and (self.n_particles > 1):
                    self.global_h += h
                if ((j==0) and (i==0)):
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
                grad_list2=np.zeros((len(self.policy_list),num_param), dtype=object)
                for n in range(len(self.policy_list)):
                    st_idx=0;length=0;
                    for param_idx in range(num_param):
                        st_idx+=length;length=origin_shape2[param_idx]
                        grad_list2[n,param_idx]=tf.reshape(tf.slice(grad[n],[st_idx],[length]),origin_shape[param_idx])
                gradients2=np.zeros(len(self.policy_list), dtype=object)
                for n in range(len(self.policy_list)):
                    gradients2[n]={};
                    for param_idx in range(num_param):
                        gradients2[n][param_keys[param_idx]]=grad_list2[n,param_idx]
                # update params
                params_list = []
                for n in range(len(self.policy_list)):
                    params_list.append(OrderedDict(zip(param_keys, [cur_params_list[i][n][key] - self.policy_list[0].step_size * gradients2[n][key] for key in param_keys])))
                new_params_list.append(params_list)
            for n, self.policy in zip(range(len(self.policy_list)), self.policy_list):
                input_list += obs_vars_list[n] + action_vars_list[n] + adv_vars_list[n] + state_info_vars_list
            if j == 0:
                init_input_list = []
                for n in range(len(self.policy_list)):
                    init_input_list += obs_vars_list[n] + action_vars_list[n] + adv_vars_list[n] + state_info_vars_list
                for n, self.policy in zip(range(len(self.policy_list)), self.policy_list):
                    self.policy.set_init_surr_obj(init_input_list, surr_objs_list[n])

        # h
        self.global_h = self.global_h / self.meta_batch_size

        ############################################################
        # reptile loss
        reptile_loss = 0.0
        for i in range(self.meta_batch_size):
            for n in range(len(self.policy_list)):
                follower_params_flat = tf.concat([tf.reshape(follower_params_list[i][n][key],[-1]) for key in param_keys],axis=0)
                global_params_flat = tf.concat([tf.reshape(global_params_list[i][n][key],[-1]) for key in param_keys],axis=0)
                reptile_loss += tf.norm(tf.stop_gradient(follower_params_flat) - global_params_flat)
        reptile_loss = reptile_loss / self.meta_batch_size
        
        for self.policy, self.optimizer in zip(self.policy_list, self.optimizer_list):
            self.optimizer.update_opt(loss=reptile_loss, target=self.policy, inputs=input_list)
        ############################################################
        
    @overrides
    def optimize_policy(self, itr, all_samples_data, particle_idx):
        logger.log("optimizing policy")
        assert len(all_samples_data) == len(self.policy_list)
        assert len(all_samples_data[0]) == self.num_leader_grad_updates
        
        input_list = []
        for step in range(len(all_samples_data[0])):
            for n in range(len(all_samples_data)):
                obs_list, action_list, adv_list = [], [], []
                for i in range(self.meta_batch_size):
                    inputs = ext.extract(
                        all_samples_data[n][step][i],
                        "observations", "actions", "advantages"
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    adv_list.append(inputs[2])
                input_list += obs_list + action_list + adv_list

        if particle_idx == 0 and (self.n_particles > 1):
            sess = tf.get_default_session()
            global_h = sess.run(self.global_h,feed_dict=dict(list(zip(self.policy_list[0].input_list_for_grad, input_list))))
            logger.record_tabular('global_h', global_h)
        
        self.optimizer_list[particle_idx].optimize(input_list)
        
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
