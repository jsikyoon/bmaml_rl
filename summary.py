import numpy as np

##########################
# BMAML

# point
prefix = 'data/local/bmaml-chaser-point100/'
file_path = 'bmaml_M2_SVPG_alpha0.1_fbs10_mbs20_flr_0.01_mlr_0.01_randomseed1/'
##########################
# EMAML

# point
#prefix = 'data/local/emaml-reptile-point100/'
#file_path = 'emaml_M2_VPG_fbs10_mbs20_flr_0.1_mlr_0.01_randomseed1/'
##########################

file_name = 'progress.csv'

class BMAMLRLSUMMARY():
    """
    result summary
    """
    def __init__(self):
        # default setting
        self.num_updates = 1
        self.global_h_list = []
        self.task_wise_summary = []
        self.particle_wise_summary =[]

    def get_summaries(self, filename, maml_case=False):

        if maml_case:
            if filename.split("/")[-2][:5] != 'emaml':
                raise ValueError  # maml_case is only working on emaml data
       
        self.get_contents(filename, maml_case)
        self.get_task_wise_summary()
        self.get_particle_wise_summary()
 
        return self.global_h_list, self.task_wise_summary, self.particle_wise_summary

    def get_contents(self, filename, maml_case=False):

        num_updates = self.num_updates

        f = open(filename, 'r')
        contents = [ line[:-1].split(',') for line in f.readlines() ]

        header = contents[0]
        contents = contents[1:]
        total_col_num = len(header)
 
        # get mbs
        filename_list = filename.split("_")
        for part_filename in filename_list:
            if 'mbs' in part_filename:
                meta_batch_size = int(part_filename[3:])

        return_indices = []
        global_h_idx = -1
        for header_part, idx in zip(header, range(len(header))):
            if 'AverageReturn' in header_part:
                if maml_case:
                    if header_part.split("_")[1] == '0':    # just using first particle
                        return_indices.append(idx)
                else:
                    return_indices.append(idx)
            elif 'global_h' in header_part:
                global_h_idx = idx

        header = np.array(header, dtype=str)

        header = header[return_indices]

        contents = np.array(contents, dtype=float)

        if global_h_idx != -1:
            global_h_list = contents[:,global_h_idx].tolist()

        contents = contents[:,return_indices]

        # this doesn't sort correctly but okay because it orders in its own particular rule
        con_idx = np.argsort(header)
        header = header[con_idx]
        contents =contents[:,con_idx]

        num_particles = int(len(header)/(meta_batch_size * (num_updates + 1)))
        
        # make class variables
        self.file_path = filename.split("/")[-2]
        self.meta_batch_size = meta_batch_size
        self.num_particles = 1 if maml_case else num_particles
        self.contents = contents
        self.global_h_list = global_h_list

    def get_task_wise_summary(self):
        contents = self.contents
        num_updates = self.num_updates
        meta_batch_size = self.meta_batch_size
        num_particles = self.num_particles
        
        reshaped_returns = []
        for c in contents:
            update_return_list = []
            for update_idx in range(num_updates + 1):
                max_return_list = []
                for batch_idx in range(meta_batch_size):
                    particle_return_list = []
                    for p_idx in range(num_particles):
                        particle_return_list.append(c[update_idx * (meta_batch_size * num_particles) + p_idx * meta_batch_size + batch_idx])
                    max_return_list.append(np.max(particle_return_list))
                update_return_list.append(np.mean(max_return_list))
            reshaped_returns.append(update_return_list)
        
        task_wise_summary = []
        for r in reshaped_returns:
            task_wise_summary.append(r[-1])

        # make class variables
        self.task_wise_summary = task_wise_summary

    def get_particle_wise_summary(self):
        contents = self.contents
        num_updates = self.num_updates
        meta_batch_size = self.meta_batch_size
        num_particles = self.num_particles
        reshaped_returns = []

        for c in contents:
            # find best particles
            update_idx = num_updates
            particle_return_list = []
            for p_idx in range(num_particles):
                particle_return_list.append(np.mean(c[(update_idx * (meta_batch_size * num_particles) + p_idx * meta_batch_size) : (update_idx * (meta_batch_size * num_particles) + (p_idx + 1) * meta_batch_size)]))
            best_p_idx = np.argmax(particle_return_list)
            # write that results
            update_return_list = []
            for update_idx in range(num_updates + 1):
                update_return_list.append(np.mean(c[(update_idx * (meta_batch_size * num_particles) + best_p_idx * meta_batch_size) : (update_idx * (meta_batch_size * num_particles) + (best_p_idx + 1) * meta_batch_size)]))
            reshaped_returns.append(update_return_list)
       
        particle_wise_summary = []
        for r in reshaped_returns:
            particle_wise_summary.append(r[-1])

        # make class variables
        self.particle_wise_summary = particle_wise_summary

if __name__=="__main__":

    filename = prefix + file_path + file_name

    bmaml_rl_summary = BMAMLRLSUMMARY()
    global_h_list, task_wise_summary, particle_wise_summary = bmaml_rl_summary.get_summaries(filename)
    file_path = bmaml_rl_summary.file_path

    # global_h
    print(file_path)
    for global_h in global_h_list:
        print(global_h)
    # task wise summary
    print(file_path)
    for tws in task_wise_summary:
        print(tws)
    # particle wise summary
    print(file_path)
    for pws in particle_wise_summary:
        print(pws)
