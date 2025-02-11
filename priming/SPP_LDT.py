import warnings

import pyactr as actr
import simpy
import re
import pandas as pd

class Model:
    """
    Model for fan experiment. We will abstract away from environment, key presses and visual module (the same is done in the abstract model of Lisp ACT-R).
    """

    def __init__(self, prime, target, **kwargs):
        env = actr.Environment(focus_position=(0, 0))
        self.model = actr.ACTRModel(environment=env, **kwargs)

        actr.chunktype("meaning", "word")
        actr.chunktype("goal", "state")

        # dict_dm = {}
        words = f"{prime} {target}".split()
        self.dm = self.model.decmem
        prime_chunk = actr.makechunk(typename="meaning", word=prime)
        self.dm.add(prime_chunk)
        target_chunk = actr.makechunk(typename="meaning", word=target)
        self.dm.add(target_chunk)
        # for word in words: #dict_dm[word]
        #     w_chunk = actr.makechunk(nameofchunk=word, typename="meaning", word=word)
        #     self.dm.add(w_chunk)
        # self.model.set_decmem(set(dict_dm.values()))
        self.dm = self.model.decmem  ###  WHY twice??

        g = self.model.goal
        g.add(actr.makechunk(nameofchunk="beginning", typename="goal", state="start"))

        self.imaginal = self.model.set_goal(name="imaginal", delay=0.2)
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
                                    state       retrieving
                                +visual>
                                    isa         _visual
                                    cmd         move_attention
                                    screen_pos  =visual_location
        """)

        #3. try to retrieve the target word from dm
        self.model.productionstring(name="retrieving", string="""
                                =g>
                                    isa         goal
                                    state       retrieving
                                =visual>
                                    isa         _visual
                                    value       =val
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
    stim = [{1: {'text': target, 'position': (150, 150)}}]

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


def experiments(mas, noise, pairs, embeddings, latency_factor):
    # env = actr.Environment(focus_position=(0, 0))
    results_df = pd.DataFrame(columns=['prime', 'target', 'predicted_rt', 'accuracy'])
    accuracy_accum = 0

    for prime, target in pairs:
        model = Model(prime=prime, target=target,
                      # environment=env,
                      automatic_visual_search=False,
                      motor_prepared=True,
                      subsymbolic=True,
                      latency_factor=latency_factor, strength_of_association=mas,
                      buffer_spreading_activation={"imaginal": 1},
                      spreading_activation_restricted=True,
                      association_only_from_chunks=False,
                      activation_trace=True, strict_harvesting=False,
                      retrieval_threshold=-2,
                      instantaneous_noise=noise, emma=False,
                      embeddings=embeddings)
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

def read_data(dataset_name):

    data = pd.read_csv(f'../data/{dataset_name}.csv', index_col=0)
    if 'prime' not in list(data.columns) or 'target' not in list(data.columns):
        assert NotImplementedError
    unique_prime_target_tuples = data[['prime', 'target']].drop_duplicates()
    pairs_list = list(unique_prime_target_tuples.itertuples(index=False, name=None))

    return pairs_list


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    mas = 1.0  # maximum association strength
    noise = 0.0
    # pairs = [('body', 'abdomen'), ('ability', 'capability')]
    dataset_name = "spp_short_rem"  # spp_short_rem for w2v
    embeddings = 'spp_w2v'  # 'spp_bert_L0'
    latency_factor = 0.7  #0.1  # default, 0.63
    pairs = read_data(dataset_name=dataset_name)
    results = experiments(mas=mas, noise=noise, pairs=pairs, embeddings=embeddings, latency_factor=latency_factor)
    results.to_csv(f'../data/results/{dataset_name}_{embeddings}_mas={mas}_lf={latency_factor}.csv')



