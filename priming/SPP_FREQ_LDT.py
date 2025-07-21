import warnings
import sys
sys.path.insert(0, "../pyactr")
import pyactr as actr
import simpy
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# import pytensor
# import pytensor.tensor as pt

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR


class Model:
    """
    Model for fan experiment. We will abstract away from environment, key presses and visual module (the same is done in the abstract model of Lisp ACT-R).
    """

    def __init__(self, prime, target, target_activation, data_path, **kwargs):
        env = actr.Environment(focus_position=(0, 0))
        self.model = actr.ACTRModel(environment=env, data_path=data_path, **kwargs)

        actr.chunktype("meaning", "word")
        actr.chunktype("goal", "state")

        # dict_dm = {}
        words = f"{prime} {target}".split()
        self.dm = self.model.decmem
        prime_chunk = actr.makechunk(typename="meaning", word=prime)
        self.dm.add(prime_chunk)
        target_chunk = actr.makechunk(typename="meaning", word=target)
        self.dm.add(target_chunk)
        activation_dict = {target_chunk: target_activation}
        self.dm.activations.update(activation_dict)
        self.dm = self.model.decmem  ###  WHY twice??

        g = self.model.goal
        g.add(actr.makechunk(nameofchunk="beginning", typename="goal", state="start"))

        self.imaginal = self.model.set_goal(name="imaginal", delay=0.2)
        self.interm = self.model.set_goal(name="interm", delay=0.2)
        self.imaginal.add(prime_chunk)

        self.env = env

        visual, visual_location = self.model.visualBuffer("visual", "visual_location",
                                                     default_harvest=self.dm, finst=1)

        # 1. find the target word in the screen
        self.model.productionstring(name="find target", string="""
                                =g>
                                    isa          goal
                                    state        start
                                ?visual_location>
                                    buffer       empty
                                ?manual>
                                    state        free
                            ==>
                                =g>
                                    isa         goal
                                    state       attend
                                ?visual_location>
                                    attended    False
                                +visual_location>
                                    isa         _visuallocation
                                    screen_x    closest        
        """)

        #2. attend the found target
        self.model.productionstring(name="attend target", string="""
                                =g>
                                    isa         goal
                                    state       attend
                                =visual_location>
                                    isa         _visuallocation
                                ?visual>
                                    state       free
                            ==>
                                =g>
                                    isa         goal
                                    state       encode
                                +visual>
                                    isa         _visual
                                    cmd         move_attention
                                    screen_pos  =visual_location
        """)
        self.model.productionstring(name="encode target", string="""
                                =g> 
                                    isa         goal
                                    state       encode
                                =visual>
                                    isa         _visual
                                    value       =val
                                ?interm>
                                    buffer      empty
                                    state       free
                            ==>
                                =g>
                                    isa         goal
                                    state       retrieving
                                +interm>
                                    isa         meaning
                                    word        =val
        
        """)

        #3. try to retrieve the target word from dm
        self.model.productionstring(name="retrieving", string="""
                                =g>
                                    isa         goal
                                    state       retrieving
                                =interm>
                                    isa         meaning
                                    word       =val
                                ?retrieval>
                                    state       free
                            ==>
                                =g>
                                    isa         goal
                                    state       retrieval_done
                                +retrieval>
                                    isa         meaning
                                    word        =val
        """)

        #4. if successful: press J
        self.model.productionstring(name="target retrieved", string="""
                                =g>
                                    isa         goal
                                    state       retrieval_done
                                ?retrieval>
                                    buffer      full
                                    state       free
                                ?manual>
                                    state       free
                            ==>
                                +manual>
                                    isa         _manual
                                    cmd         press_key
                                    key         J
                                =g>
                                    isa         goal
                                    state       start
                                ~visual>
                                ~visual_location>
        """)

        # self.model.productionstring(name="encode target", string="""
        #                         =g>
        #                             isa         goal
        #                             state       encoding
        #                         =retrieval>
        #                             isa         meaning
        #                             word        =val
        #                         ?imaginal>
        #                             buffer      empty
        #                             state       free
        #                     ==>
        #                         +imaginal>
        #                             isa         meaning
        #                             word        =val
        #                         =g>
        #                             isa         goal
        #                             state       start
        # """)

        #5. if failed: press F
        self.model.productionstring(name="target not found", string="""
                                =g>
                                    isa         goal
                                    state       retrieval_done
                                ?retrieval>
                                    state       error
                                ?manual>
                                    state       free
                            ==>
                                +manual>
                                    isa         _manual
                                    cmd         press_key
                                    key         F
                                =g>
                                    isa         goal
                                    state       start
                                ~visual>
                                ~visual_location>
        """)


def run_simulation(env, model, target):
    stim = [{1: {'text': target, 'position': (150, 150), 'vis_delay':len(target)}}] #vis_delay=n_char(target)

    # run new simulation
    sim = model.model.simulation(realtime=False, gui=False,
                           environment_process=env.environment_process,
                           stimuli=stim, triggers=[['J', 'F']], times=30, trace=True
                           )
    key = ""
    rt = 0
    while key == "":
        try:
            sim.step()
        except simpy.core.EmptySchedule:
            break
        if re.search("^RULE FIRED:", str(sim.current_event.action)):
            continue
        if re.search("^RETRIEVED: None", str(sim.current_event.action)):
            continue
        if re.search("^KEY PRESSED:", str(sim.current_event.action)):
            key = re.search(r".$", str(sim.current_event.action)).group()
            rt = sim.show_time()
    # response = key == "J"
    return rt, key


def experiments(mas, noise, pairs, neigh, fan, embeddings, latency_factor, latency_exponent, decay, activations, 
                spp_bins, data_path, emma):
    # env = actr.Environment(focus_position=(0, 0))
    results_df = pd.DataFrame(columns=['prime', 'target', 'predicted_rt', 'accuracy'])
    accuracy_accum = 0
    neigh_cos, num_fan = None, None
    for elem in pairs:
        # prime, target, neigh_cos = None, None, None
        if neigh:
            prime, target, neigh_cos = elem[0], elem[1], elem[2]
        elif fan:
            prime, target, num_fan = elem[0], elem[1], elem[2]
        else:
            prime, target = elem[0], elem[1]
        print(f'prime: {prime}, target: {target}, len(target): {len(target)}')
        bin = spp_bins[(spp_bins['prime'] == prime) & (spp_bins['target'] == target)]['bin_index'].item()
        target_activation = activations[bin]

        print(f'target activation for bin={bin} is {target_activation}')
        model = Model(prime=prime, target=target,
                      data_path=data_path,
                      target_activation=target_activation,
                      # environment=env,
                      automatic_visual_search=False,
                      motor_prepared=True,
                      subsymbolic=True,
                      latency_factor=latency_factor,
                      latency_exponent=latency_exponent,
                      decay=decay,
                      strength_of_association=mas,
                      buffer_spreading_activation={"imaginal": 1},
                      spreading_activation_restricted=True,
                      association_only_from_chunks=False,
                      activation_trace=True,
                      strict_harvesting=False,
                      retrieval_threshold=-80,
                      instantaneous_noise=noise,
                      embeddings=embeddings,
                      neigh_cos=neigh_cos,
                      fan=num_fan,
                      emma=emma, 
                      emma_noise=False)
        env = model.env
        rt, response = run_simulation(env=env, model=model, target=target)
        accuracy = response == "J"  # change if we have non-word targets
        accuracy_accum += accuracy
        print(f"Accuracy for ({prime},{target}) = {float(accuracy)}")
        print(f"Reading time for the target `{target}` is {rt * 1000} (ms)")
        # add results to df
        data = [{'prime': prime, 'target': target, 'predicted_rt': rt, 'accuracy': accuracy}]
        results_df = pd.concat([results_df, pd.DataFrame(data)], ignore_index=True)

    return results_df


def get_activations(FREQ):

    def time_freq(freq):
        rehearsals = np.zeros((np.max(freq).astype(int) * 113, len(freq)))
        print(f'rehearsals.shape={rehearsals.shape}')
        for i in np.arange(len(freq)):
            temp = np.arange((freq[i] * 112.5)).astype(int)
            temp = temp * np.array(SEC_IN_TIME / (freq[i] * 112.5)).astype(int)
            rehearsals[:len(temp), i] = temp
        return rehearsals.T

    time = time_freq(FREQ)
    scaled_time = time ** (-decay)

    def compute_activation(scaled_time_vector):
        subvector = scaled_time_vector[~np.isinf(scaled_time_vector)]
        return np.log(subvector.sum())

    activation_from_time = [ compute_activation(row) for row in scaled_time]
    print('activation_from_time:', activation_from_time)
    # activation_from_time = [-9.21181007, -8.93834678, -8.32109255, -8.173624, -7.9042454, -7.3922061,
    #  -7.19796136, -7.07670266, -6.66561281, -6.49680178, -6.30419527, -6.03986274,
    #  -5.72818052, -5.53672519, -5.1546272, -5.03540494]
    activations = {f'q{i}': activation_from_time[i] for i in range(max_bin)}  # only q0-qmax
    print("activations:", activations)
    return activations


def read_data(dataset_name):

    data = pd.read_csv(f'../data/{dataset_name}.csv', index_col=0)
    if 'prime' not in list(data.columns) or 'target' not in list(data.columns):
        assert NotImplementedError
    cols = ['prime', 'target']
    neigh, fan = False, False
    if 'avg_neigh_cos' in list(data.columns):
        cols.append('avg_neigh_cos')
        neigh = True
    elif 'fan' in list(data.columns):
        cols.append('fan')
        fan = True
    unique_prime_target_tuples = data[cols].drop_duplicates()
    pairs_list = list(unique_prime_target_tuples.itertuples(index=False, name=None))

    return pairs_list, neigh, fan


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    mas = 4.0  # maximum association strength
    noise = 0.0
    dataset_name = "spp_short_neigh" #"spp_short_fan" #"spp_short_neigh" #"spp_short_rem"  # spp_short_rem for w2v
    embeddings = 'spp_bert_L1_std' #'spp_bert_L1_std' #'spp_w2v'  # 'spp_bert_L0'
    latency_factor = 0.379287  # default, 0.63
    lateny_exponent = 0.363791  # 1.0
    decay = 0.153496  # 0.5
    max_bin = 100
    emma = True
    data_path = "../data"
    pairs, neigh, fan = read_data(dataset_name=dataset_name)
    # data_path = args.data_path
    # Freq input
    # spp_freq = pd.read_csv(f'{data_path}/spp_freq.csv')
    
    spp_bins = pd.read_csv(f'{data_path}/spp_bins{str(max_bin)}.csv')
    if neigh or fan:
        # Get the set of primes that exist in spp
        unique_primes = set([elem[0] for elem in pairs]) # all unique primes
        # Filter spp_short to keep only rows where the prime is in the set of primes from spp
        spp_bins = spp_bins[spp_bins['prime'].isin(unique_primes)].copy()

    mean_freqs_sorted = {f'q{i}': spp_bins[spp_bins['bin_index'] == f'q{i}']['mean_freq'].iloc[0] for i in range(max_bin)} #mean_freq
    FREQ = list(mean_freqs_sorted.values())
    # FREQ = np.array(spp_bins['mean_freq'])
    activations = get_activations(FREQ)
    results = experiments(mas=mas, noise=noise, pairs=pairs, neigh=neigh, fan=fan,
                          embeddings=embeddings,
                          latency_factor=latency_factor,
                          latency_exponent=lateny_exponent,
                          decay=decay,
                          activations=activations,
                          spp_bins=spp_bins,
                          data_path=data_path,
                          emma=emma)

    real_rt = spp_bins['target.RT'].to_list()
    pred_rt = results['predicted_rt'].to_list()
    pred_rt = [rt * 1000 for rt in pred_rt]
    print(f'spearmans correlation for {dataset_name}_{embeddings}_mas={mas}_lf={latency_factor}_le={lateny_exponent}_decay={decay}_emma={emma} = ', spearmanr(real_rt, pred_rt)[0])
    results.to_csv(f'{data_path}/results/{dataset_name}_{embeddings}_mas={mas}_lf={latency_factor}_le={lateny_exponent}_decay={decay}_emma={emma}.csv')



