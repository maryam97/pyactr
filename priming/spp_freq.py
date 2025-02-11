"""
A model of lexical decision: Bayes+ACT-R, with imaginal buffer;
default delay for the imaginal buffer (200 ms)
"""

import warnings

import pandas as pd
import pyactr as actr
import numpy as np
import pymc as pm
import arviz as az
from pymc import Normal, HalfNormal, Deterministic, Uniform
import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import argparse
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
# on average, 15 years of exposure is 112.5 million words

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR

class Model:
    """
    Model for fan experiment. We will abstract away from environment, key presses and visual module (the same is done in the abstract model of Lisp ACT-R).
    """

    def __init__(self, model):
        # env = actr.Environment(focus_position=(0, 0))
        # self.model = actr.ACTRModel(environment=env, **kwargs)
        self.model = model

        actr.chunktype("meaning", "word")
        actr.chunktype("goal", "state")

        # dict_dm = {}
        # words = f"{prime} {target}".split()
        self.dm = self.model.decmem
        # prime_chunk = actr.makechunk(typename="meaning", word=prime)
        # self.dm.add(prime_chunk)
        # target_chunk = actr.makechunk(typename="meaning", word=target)
        # self.dm.add(target_chunk)

        self.g = self.model.goal
        # self.g.add(actr.makechunk(nameofchunk="beginning", typename="goal", state="start"))

        self.imaginal = self.model.set_goal(name="imaginal", delay=0.2)
        # self.imaginal.add(prime_chunk)

        # self.env = env

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


def run_stimulus(model, env, target, prime):
    """
    Function running one instance of lexical decision for a word.
    """
    # reset model state to initial state for a new simulation
    # (flush buffers without moving their contents to dec mem)
    try:
        model.retrieval.pop()
    except KeyError:
        pass
    try:
        model.g.pop()
        # model.goals["g"].pop()
    except KeyError:
        pass
    try:
        model.imaginal.pop()
        # model.goals["imaginal"].pop()
    except KeyError:
        pass

    # reinitialize model
    stim = {1: {'text': target, 'position': (320, 180)}}
    # model.goals["g"].add(actr.makechunk(nameofchunk='start',
    #                                            typename="goal",
    #                                            state='attend'))
    # model.goals["imaginal"].add(actr.makechunk(nameofchunk='start',
    #                                                   typename="word"))
    # model.goals["imaginal"].delay = 0.2

    prime_chunk = actr.makechunk(typename="meaning", word=prime)
    model.dm.add(prime_chunk)
    target_chunk = actr.makechunk(typename="meaning", word=target)
    model.dm.add(target_chunk)

    # self.g = self.model.goal
    model.g.add(actr.makechunk(nameofchunk="beginning", typename="goal", state="start"))

    # self.imaginal = self.model.set_goal(name="imaginal", delay=0.2)
    model.imaginal.delay = 0.2
    model.imaginal.add(prime_chunk)

    env.current_focus = [320, 180]
    model.model_parameters['motor_prepared'] = True

    # run new simulation; switch to gui=True to suppress pyactr output when estimating Bayesian model
    lex_dec_sim = model.simulation(realtime=False, gui=False, trace=False,
              environment_process=env.environment_process,
              stimuli=stim, triggers='', times=10)
    while True:
        lex_dec_sim.step()
        if lex_dec_sim.current_event.action == "KEY PRESSED: J":
            estimated_time = lex_dec_sim.show_time()
            break
        if lex_dec_sim.current_event.action == "KEY PRESSED: F":
            estimated_time = -1
            break
    return estimated_time


def run_lex_decision_task(model, env, pairs):
    """
    Function running a full lexical decision task:
    it calls run_stimulus(word) for words from all 16 freq bands.
    """
    sample = []
    # for word in ORDERED_FREQ:
    for target, prime in pairs:
        sample.append(run_stimulus(model=model, env=env, target=target, prime=prime))
    return sample


@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dvector],
       otypes=[pt.dvector])
def actrmodel_latency(model, env, pairs, lf, le, decay, activation_from_time):
    """
    Function running the entire lexical decision task for specific
    values of the latency factor, latency exponent and decay parameters.
    The activation computed with the specific value of the decay
    parameter is also inherited as a separate argument to save expensive
    computation time.
    The function is wrapped inside the theano @as_op decorator so that
    pymc3 / theano can use it as part of the RT likelihood function in the
    Bayesian model below.
    """
    model.model_parameters["latency_factor"] = np.array(lf).astype("float32").item()
    model.model_parameters["latency_exponent"] = np.array(le).astype("float32").item()
    model.model_parameters["decay"] = np.array(decay).astype("float32").item()
    activation_dict = {x[0]: np.array(x[1]).astype("float32").item()
                       for x in zip(LEMMA_CHUNKS, activation_from_time)}
    model.decmem.activations.update(activation_dict)
    sample = run_lex_decision_task(model=model, env=env, pairs=pairs)
    return np.array(sample)


# def experiments(mas, noise, pairs, embeddings, latency_factor):
#     # env = actr.Environment(focus_position=(0, 0))
#     # results_df = pd.DataFrame(columns=['prime', 'target', 'predicted_rt', 'accuracy'])
#     accuracy_accum = 0
#     actr_env = actr.Environment(focus_position=(320, 180))
#     actr_model = actr.ACTRModel(environment=actr_env, automatic_visual_search=False,
#                           motor_prepared=True,
#                           subsymbolic=True,
#                           latency_factor=latency_factor,
#                           strength_of_association=mas,
#                           buffer_spreading_activation={"imaginal": 1},
#                           spreading_activation_restricted=True,
#                           association_only_from_chunks=False,
#                           activation_trace=True, strict_harvesting=False,
#                           retrieval_threshold=-2,
#                           instantaneous_noise=noise, emma=False,
#                           embeddings=embeddings)
#
#     # for prime, target in pairs:
#     model = Model(model=actr_model)
#         # env = model.env
#         # rt, response = run_simulation(env=env, model=model, target=target)
#         # accuracy = response == "J"  # change if we have non-word targets
#         # accuracy_accum += accuracy
#         # print(f"Accuracy for ({prime},{target}) = {float(accuracy)}")
#         # print(f"Reading time for the target `{target}` is {rt * 1000} (ms)")
#         # # add results to df
#         # data = [{'prime': prime, 'target': target, 'predicted_rt': rt, 'accuracy': accuracy}]
#         # results_df = pd.concat([results_df, pd.DataFrame(data)], ignore_index=True)
#
#     return #model #results_df

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        default='.')
    parser.add_argument('--chains',
                        type=int,
                        default=1)
    parser.add_argument('--draws',
                        type=int,
                        default=10)
    parser.add_argument('--tunes',
                        type=int,
                        default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    mas = 1.0  # maximum association strength
    noise = 0.0

    dataset_name = "spp_short_rem"  # spp_short_rem for w2v
    embeddings = 'spp_w2v'  # 'spp_bert_L0'
    # latency_factor = 0.7  #0.1  # default, 0.63
    # pairs = read_data(dataset_name=dataset_name)
    # Freq input
    spp_freq = pd.read_csv('../data/spp_freq.csv')
    FREQ = np.array(spp_freq['mean_freq'])
    RT = np.array(spp_freq['target_rt']) / 1000
    ACCURACY = np.ones(spp_freq.shape[0])

    FREQ_DICT = {}
    for i in range(spp_freq.shape[0]):
        row = spp_freq.iloc[i]
        FREQ_DICT[spp_freq.iloc[i].target] = spp_freq.iloc[i].mean_freq * 112.5

    # ORDERED_FREQ = sorted(list(FREQ_DICT), key=lambda x: FREQ_DICT[x], reverse=True)

    def time_freq(freq):
        rehearsals = np.zeros((np.max(freq).astype(int) * 113, len(freq)))
        for i in np.arange(len(freq)):
            temp = np.arange((freq[i] * 112.5)).astype(int)
            temp = temp * np.array(SEC_IN_TIME / (freq[i] * 112.5)).astype(int)
            rehearsals[:len(temp), i] = temp
        return rehearsals.T

    time = time_freq(FREQ)

    # results = experiments(mas=mas, noise=noise, pairs=pairs, embeddings=embeddings, latency_factor=latency_factor)
    # results.to_csv(f'../data/results/{dataset_name}_{embeddings}_mas={mas}_lf={latency_factor}.csv')
    actr_env = actr.Environment(focus_position=(320, 180))
    actr_model = actr.ACTRModel(environment=actr_env, automatic_visual_search=False,
                                motor_prepared=True,
                                subsymbolic=True,
                                # latency_factor=latency_factor,
                                strength_of_association=mas,
                                buffer_spreading_activation={"imaginal": 1},
                                spreading_activation_restricted=True,
                                association_only_from_chunks=False,
                                activation_trace=True, strict_harvesting=False,
                                retrieval_threshold=-2,
                                instantaneous_noise=noise, emma=False,
                                embeddings=embeddings)

    # for prime, target in pairs:
    prime_model = Model(model=actr_model)
    LEMMA_CHUNKS = [(actr.makechunk("", typename="word", form=word)) ###???
                    for word in spp_freq.target] #ORDERED_FREQ, already sorted
    prime_model.model.set_decmem({x: np.array([]) for x in LEMMA_CHUNKS}) ###?
    pairs = []
    for i in range(spp_freq.shape[0]):  # add in order of frequencies
        pairs.append((spp_freq.iloc[i].target, spp_freq.iloc[i].prime))

    # Bayesian Model
    lex_decision_with_bayes = pm.Model()
    with lex_decision_with_bayes:
        # prior for activation
        decay = Uniform('decay', lower=0, upper=1)
        # priors for accuracy
        noise = Uniform('noise', lower=0, upper=5)
        threshold = Normal('threshold', mu=0, sigma=10)
        # priors for latency
        lf = HalfNormal('lf', sigma=1)
        le = HalfNormal('le', sigma=1)
        # compute activation
        scaled_time = time ** (-decay)

        def compute_activation(scaled_time_vector):
            compare = pt.isinf(scaled_time_vector)
            subvector = scaled_time_vector[(1 - compare).nonzero()]
            activation_from_time = pt.log(subvector.sum())
            return activation_from_time

        activation_from_time, _ = pytensor.scan(fn=compute_activation, sequences=scaled_time)
        # latency likelihood -- this is where pyactr is used
        pyactr_rt = actrmodel_latency(model=prime_model.model, env=actr_env, pairs=pairs, lf=lf, le=le, decay=decay,
                                      activation_from_time=activation_from_time)
        mu_rt = Deterministic('mu_rt', pyactr_rt)
        rt_observed = Normal('rt_observed', mu=mu_rt, sigma=0.01, observed=RT)
        # accuracy likelihood
        odds_reciprocal = pt.exp(-(activation_from_time - threshold) / noise)
        mu_prob = Deterministic('mu_prob', 1 / (1 + odds_reciprocal))
        prob_observed = Normal('prob_observed', mu=mu_prob, sigma=0.01, observed=ACCURACY)

    with lex_decision_with_bayes:
        num_draws = args.draws  # 1000
        num_chains = args.chains  # 4
        num_tunes = args.tunes  # 10000

        step = pm.DEMetropolisZ(tune="scaling", proposal_dist=pm.NormalProposal)
        trace = pm.sample(draws=num_draws, tune=num_tunes, chains=num_chains, step=step)

        print('trace=', trace)
        print('saving trace...')
        trace.to_netcdf(f'{args.root}/trace_draws={num_draws}_tune={num_tunes}_chains={num_chains}.nc')

