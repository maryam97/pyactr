import warnings

import pyactr as actr
import simpy
import re

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
        self.dm = self.model.decmem

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


def experiments(mas, noise, pairs):
    # env = actr.Environment(focus_position=(0, 0))
    accuracy_accum = 0

    for prime, target in pairs:
        model = Model(prime=prime, target=target,
                      # environment=env,
                      automatic_visual_search=False,
                      motor_prepared=True,
                      subsymbolic=True,
                      latency_factor=0.63, strength_of_association=mas,
                      buffer_spreading_activation={"imaginal": 1},
                      spreading_activation_restricted=True,
                      association_only_from_chunks=False,
                      activation_trace=True, strict_harvesting=False,
                      retrieval_threshold=-2,
                      instantaneous_noise=noise, emma=False,
                      embeddings='spp_w2v')
        env = model.env
        rt, response = run_simulation(env=env, model=model, target=target)
        accuracy = response == "J"  # change if we have non-word targets
        accuracy_accum += accuracy
        print(f"Accuracy for ({prime},{target}) = {float(accuracy)}")
        print(f"Reading time for the target `{target}` is {rt * 1000} (ms)")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    mas = 1.6  # maximum association strength
    noise = 0.0
    pairs = [('body', 'abdomen'), ('ability', 'capability')]
    experiments(mas=mas, noise=noise, pairs=pairs)



