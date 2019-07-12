import os
import numpy as np
from summary import BMAMLRLSUMMARY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#plt_style = 'ggplot'
plt_style = 'seaborn-paper'
plt.style.use(plt_style)

"""
Ploting MAML, EMAML and BMAML for different number of particles
"""

##########################
# params
prefix = 'data/local/'
file_name = 'progress.csv'
out_folder = 'plots'
env_type = 'point100' 
bmaml_meta_method = 'trpo' # {trpo|chaser}
emaml_meta_method = 'trpo' # {trpo|reptile}
svpg_alpha = 1.0
fbs = 10
mbs = 20
flr = 0.1
svpg_flr = 0.1
mlr = 0.01 
num_particles = [5]
##########################

class BMAMLRLPLOT():
    """
    result plot
    """
    def __init__(self, bmaml_rl_summary):
        self.brs = bmaml_rl_summary
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

    def make_plot(self):
        self.get_paths()
        self.plotting()

    def get_paths(self):
        
        # file path and prefix
        sub_prefix = '_fbs' + str(fbs) + '_mbs' + str(mbs) + '_flr_' + str(svpg_flr) + '_mlr_' + str(mlr)
        bmaml_path = os.path.join(prefix,'bmaml-'+str(bmaml_meta_method)+'-'+env_type)
        bmaml_prefix_list = []
        for num_p in num_particles:
            bmaml_prefix_list.append('bmaml_M'+str(num_p) + '_SVPG_alpha' + str(svpg_alpha) + sub_prefix + '_randomseed')
        sub_prefix = '_fbs' + str(fbs) + '_mbs' + str(mbs) + '_flr_' + str(flr) + '_mlr_' + str(mlr)
        emaml_path = os.path.join(prefix,'emaml-'+str(emaml_meta_method)+'-'+env_type)
        emaml_prefix_list = []
        for num_p in num_particles:
            emaml_prefix_list.append('emaml_M'+str(num_p) + '_VPG' + sub_prefix + '_randomseed')

        # find random seed and get file name
        bmaml_list = os.listdir(bmaml_path)
        bmaml_filename_list, bmaml_randseed_list = [], []
        for bmaml_prefix in bmaml_prefix_list:
            bfl, brl = [], []
            for bmaml_folder in bmaml_list:
                if bmaml_prefix in bmaml_folder:
                    brl.append(int(bmaml_folder.split("randomseed")[1]))
                    bfl.append(os.path.join(bmaml_path, bmaml_folder, file_name))
            bmaml_filename_list.append(bfl)
            bmaml_randseed_list.append(brl)

        for idx in range(len(num_particles)):
            bmaml_randseed_list[idx].sort()
        
        emaml_list = os.listdir(emaml_path)
        emaml_filename_list, emaml_randseed_list = [], []
        for emaml_prefix in emaml_prefix_list:
            efl, erl = [], []
            for emaml_folder in emaml_list:
                if emaml_prefix in emaml_folder:
                    erl.append(int(emaml_folder.split("randomseed")[1]))
                    efl.append(os.path.join(emaml_path, emaml_folder, file_name))
            emaml_filename_list.append(efl)
            emaml_randseed_list.append(erl)

        for idx in range(len(num_particles)):
            emaml_randseed_list[idx].sort()

        try:
            randseeds = bmaml_randseed_list[0]
            for b_idx in range(1, len(bmaml_randseed_list)):
                for r_idx in range(len(bmaml_randseed_list[0])):
                    if(randseeds[r_idx] != bmaml_randseed_list[b_idx][r_idx]):
                        raise ValueError # not fair comparison
            for e_idx in range(len(emaml_randseed_list)):
                for r_idx in range(len(emaml_randseed_list[0])):
                    if(randseeds[r_idx] != emaml_randseed_list[e_idx][r_idx]):
                        raise ValueError # not fair comparison
        except:
            raise ValueError # not fair comparison

        print(bmaml_filename_list)
        print(emaml_filename_list)
        # make class variables
        self.sub_prefix = sub_prefix
        self.bmaml_filename_list = bmaml_filename_list
        self.emaml_filename_list = emaml_filename_list

    def plotting(self):
        brs = self.brs  # BMAMLRLSUMMARY
        sub_prefix = self.sub_prefix
        emaml_filename_list = self.emaml_filename_list
        bmaml_filename_list = self.bmaml_filename_list

        # get summaries
        e_gh, e_tws, e_pws = [], [], []
        m_tws, m_pws = [], []
        flags = 0
        for efl in emaml_filename_list:
            e_gh_p, e_tws_p, e_pws_p = [], [], []
            for e_fn in efl:
                global_h, tws, pws = brs.get_summaries(e_fn)
                e_gh_p.append(global_h)
                e_tws_p.append(tws)
                e_pws_p.append(pws)
                if flags == 0:
                    _, tws, pws = brs.get_summaries(e_fn, maml_case=True)
                    m_tws.append(tws)
                    m_pws.append(pws)
            flags = 1
            e_gh.append(e_gh_p)
            e_tws.append(e_tws_p)
            e_pws.append(e_pws_p)
        
        b_gh, b_tws, b_pws = [], [], []
        for bfl in bmaml_filename_list:
            b_gh_p, b_tws_p, b_pws_p = [], [], []
            for b_fn in bfl:
                global_h, tws, pws = brs.get_summaries(b_fn)
                b_gh_p.append(global_h)
                b_tws_p.append(tws)
                b_pws_p.append(pws)
            b_gh.append(b_gh_p)
            b_tws.append(b_tws_p)
            b_pws.append(b_pws_p)

        # ploting
        e_gh = np.array(e_gh)
        e_tws = np.array(e_tws)
        e_pws = np.array(e_pws)
        m_gh = []
        m_tws = np.array(m_tws)
        m_pws = np.array(m_pws)
        b_gh = np.array(b_gh)
        b_tws = np.array(b_tws)
        b_pws = np.array(b_pws)

        datanamelist = ['global_h', 'tws', 'pws']
        e_datalist = [e_gh, e_tws, e_pws]
        m_datalist = [m_gh, m_tws, m_pws]
        b_datalist = [b_gh, b_tws, b_pws]
        for dn, e_d, m_d, b_d in zip(datanamelist, e_datalist, m_datalist, b_datalist):
            e_mean, e_std = [], [] 
            for p_idx in range(len(e_d)):
                e_mean_p, e_std_p = [], []
                for i in range(len(e_d[p_idx][0])):
                    e_mean_p.append(np.mean(e_d[p_idx][:,i]))
                    e_std_p.append(np.std(e_d[p_idx][:,i])/2.0)
                e_mean.append(e_mean_p)
                e_std.append(e_std_p)
            e_mean = np.array(e_mean)
            e_std = np.array(e_std)

            if dn != 'global_h':
                m_mean, m_std = [], []
                for i in range(len(m_d[0])):
                    m_mean.append(np.mean(m_d[:,i]))
                    m_std.append(np.std(m_d[:,i])/2.0)
            else:
                m_mean = e_mean
                m_std = e_std
            m_mean = np.array(m_mean)
            m_std = np.array(m_std)

            b_mean, b_std = [], [] 
            for p_idx in range(len(b_d)):
                b_mean_p, b_std_p = [], []
                for i in range(len(b_d[p_idx][0])):
                    b_mean_p.append(np.mean(b_d[p_idx][:,i]))
                    b_std_p.append(np.std(b_d[p_idx][:,i])/2.0)
                b_mean.append(b_mean_p)
                b_std.append(b_std_p)
            b_mean = np.array(b_mean)
            b_std = np.array(b_std)

            plt.subplot('111')

            e_color = ['m', 'darkred', 'darksalmon']
            for p_idx, e_m, e_s in zip(range(len(e_mean)), e_mean, e_std):
                plt.fill_between(range(1,len(e_m)+1), e_m - e_s, e_m + e_s, color=e_color[p_idx], alpha=0.3)
            if dn != 'global_h':    # maml doesn't have h, which shows an average distance between particles
                plt.fill_between(range(1,len(m_mean)+1), m_mean - m_std, m_mean + m_std, color='skyblue', alpha=0.3)
            b_color = ['c', 'y', 'gold']
            for p_idx, b_m, b_s in zip(range(len(b_mean)), b_mean, b_std):
                plt.fill_between(range(1,len(b_m)+1), b_m - b_s, b_m + b_s, color=b_color[p_idx], alpha=0.3)

            if dn != 'global_h':
                plt.plot(range(1,len(m_mean)+1), m_mean, color='b', label='MAML')
            e_color2 = ['r', 'maroon', 'salmon']
            for p_idx, e_m in zip(range(len(e_mean)), e_mean):
                plt.plot(range(1,len(e_m)+1), e_m, color=e_color2[p_idx], label='EMAML-'+emaml_meta_method.upper()+'(M='+str(num_particles[p_idx])+')')
            b_color2 = ['g', 'olive', 'goldenrod']
            for p_idx, b_m in zip(range(len(b_mean)), b_mean):
                plt.plot(range(1,len(b_m)+1), b_m, color=b_color2[p_idx], label='BMAML-'+bmaml_meta_method.upper()+'(M='+str(num_particles[p_idx])+')')
            plt.legend(bbox_to_anchor=(0.55, 0.43), loc=2, ncol=1, fontsize=12)
            axes = plt.gca()

            plt.xlabel('Meta iterations',fontsize=15)
            if dn != 'global_h':
                plt.ylabel('Average Returns',fontsize=15)
            else:
                plt.ylabel('Average Dist. between Particles',fontsize=15)
            plt.grid(True)

            # for better ploting
            axes.set_xlim([0, 100]) 

            if dn == 'global_h':
                if emaml_meta_method == 'trpo' and bmaml_meta_method == 'trpo':
                    axes.set_ylim([0, 15])  # trpo
                elif emaml_meta_method == 'reptile' and bmaml_meta_method == 'chaser':
                    axes.set_ylim([0, 15])  # chaser reptile
                else:
                    axes.set_ylim([0, 15])  # default
            else:
                axes.set_ylim([-80, 0])

            # title and naming
            title = '2D Navigation'

            outfilename_prefix = bmaml_meta_method+'_vs_'+emaml_meta_method+'_'

            if dn == 'global_h':
                title += ', average distance'
                outfilename = outfilename_prefix + env_type + sub_prefix + '_global_h_'+plt_style+'.png'
            elif dn == 'tws':
                #title += ', summary1'
                outfilename = outfilename_prefix + env_type + sub_prefix + '_summary1_'+plt_style+'.png'  # main report results
            elif dn == 'pws':
                title += ', summary2'
                outfilename = outfilename_prefix + env_type + sub_prefix + '_summary2_'+plt_style+'.png'
            else:
                raise ValueError  # not in the cases
            
            plt.title(title, fontsize=16, y=1.0)
            plt.savefig(os.path.join(out_folder,outfilename))
            plt.close()

if __name__=="__main__":

    bmaml_rl_summary = BMAMLRLSUMMARY()
    bmaml_rl_plot = BMAMLRLPLOT(bmaml_rl_summary)
    bmaml_rl_plot.make_plot()
