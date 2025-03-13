"""
A model of lexical decision: Bayes+ACT-R, with imaginal buffer;
default delay for the imaginal buffer (200 ms)
"""

import warnings
import sys
import os

import matplotlib as mpl
# mpl.use("pgf")
# pgf_with_pdflatex = {"text.usetex": True, "pgf.texsystem": "pdflatex",
                     # "pgf.preamble": [r"\usepackage{mathpazo}",
                                      # r"\usepackage[utf8x]{inputenc}",
                                      # r"\usepackage[T1]{fontenc}",
                                      # r"\usepackage{amsmath}"],
                     # "axes.labelsize": 8,
                     # "font.family": "serif",
                     # "font.serif":["Palatino"],
                     # "font.size": 8,
                     # "legend.fontsize": 8,
                     # "xtick.labelsize": 8,
                     # "ytick.labelsize": 8}
# mpl.rcParams.update(pgf_with_pdflatex)
import matplotlib.pyplot as plt
# plt.style.use('seaborn')
import seaborn as sns
# sns.set_style({"font.family":"serif", "font.serif":["Palatino"]})

import pandas as pd
import pyactr as actr
import math
from simpy.core import EmptySchedule
import numpy as np
import re
import scipy.stats as stats
import scipy

import pymc as pm
import arviz as az
from pymc import Gamma, Normal, HalfNormal, Deterministic, Uniform
# , find_MAP,Slice, sample, , Metropolis, traceplot, gelman_rubin

# from pymc.backends.base import merge_traces
# from pymc3.backends import Text
# from pymc.backends.text import load
# from pymc3.backends.text import dump

import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import argparse
warnings.filterwarnings("ignore")

# os.mknod("./data/newfile.txt")

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

FREQ = np.array([242, 92.8, 57.7, 40.5, 30.6, 23.4, 19,
                 16, 13.4, 11.5, 10, 9, 7, 5, 3, 1])
RT = np.array([542, 555, 566, 562, 570, 569, 577, 587,
               592, 605, 603, 575, 620, 607, 622, 674])/1000
ACCURACY = np.array([97.22, 95.56, 95.56, 96.3, 96.11, 94.26,
                     95, 92.41, 91.67, 93.52, 91.85, 93.52,
                     91.48, 90.93, 84.44, 74.63])/100

environment = actr.Environment(focus_position=(320, 180))
lex_decision = actr.ACTRModel(environment=environment,
                       subsymbolic=True,
                       automatic_visual_search=True,
                       activation_trace=False,
                       retrieval_threshold=-80,
                       motor_prepared=True,
                       eye_mvt_scaling_parameter=0.18,
                       emma_noise=False)

actr.chunktype("goal", "state")
actr.chunktype("word", "form")

# on average, 15 years of exposure is 112.5 million words

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR

FREQ_DICT = {}
FREQ_DICT['guy'] = 242*112.5
FREQ_DICT['somebody'] = 92*112.5
FREQ_DICT['extend'] = 58*112.5
FREQ_DICT['dance'] = 40.5*112.5
FREQ_DICT['shape'] = 30.6*112.5
FREQ_DICT['besides'] = 23.4*112.5
FREQ_DICT['fit'] = 19*112.5
FREQ_DICT['dedicate'] = 16*112.5
FREQ_DICT['robot'] = 13.4*112.5
FREQ_DICT['tile'] = 11.5*112.5
FREQ_DICT['between'] = 10*112.5
FREQ_DICT['precedent'] = 9*112.5
FREQ_DICT['wrestle'] = 7*112.5
FREQ_DICT['resonate'] = 5*112.5
FREQ_DICT['seated'] = 3*112.5
FREQ_DICT['habitually'] = 1*112.5

ORDERED_FREQ = sorted(list(FREQ_DICT), key=lambda x:FREQ_DICT[x], reverse=True)

def time_freq(freq):
    rehearsals = np.zeros((np.max(freq).astype(int) * 113, len(freq)))
    for i in np.arange(len(freq)):
        temp = np.arange((freq[i]*112.5)).astype(int)
        temp = temp * np.array(SEC_IN_TIME/(freq[i]*112.5)).astype(int)
        rehearsals[:len(temp),i] = temp
    return rehearsals.T

# time = theano.shared(time_freq(FREQ), 'time')
time=time_freq(FREQ)
print('time=', time)
# time = pt.tensor(time_freq(FREQ))
# print('time tensor=', time)

LEMMA_CHUNKS = [(actr.makechunk("", typename="word", form=word))
                for word in ORDERED_FREQ]
lex_decision.set_decmem({x: np.array([]) for x in LEMMA_CHUNKS})

lex_decision.goals = {}
lex_decision.set_goal("g")
lex_decision.set_goal("imaginal")

lex_decision.productionstring(name="attend word", string="""
    =g>
    isa     goal
    state   'attend'
    =visual_location>
    isa    _visuallocation
    ?visual>
    state   free
    ==>
    =g>
    isa     goal
    state   'encoding'
    +visual>
    isa     _visual
    cmd     move_attention
    screen_pos =visual_location
    ~visual_location>
""")

lex_decision.productionstring(name="encoding word", string="""
    =g>
    isa     goal
    state   'encoding'
    =visual>
    isa     _visual
    value   =val
    ==>
    =g>
    isa     goal
    state   'retrieving'
    +imaginal>
    isa     word
    form    =val
""")

lex_decision.productionstring(name="retrieving", string="""
    =g>
    isa     goal
    state   'retrieving'
    =imaginal>
    isa     word
    form    =val
    ==>
    =g>
    isa     goal
    state   'retrieval_done'
    +retrieval>
    isa     word
    form    =val
""")

lex_decision.productionstring(name="lexeme retrieved", string="""
    =g>
    isa     goal
    state   'retrieval_done'
    ?retrieval>
    buffer  full
    state   free
    ==>
    =g>
    isa     goal
    state   'done'
    +manual>
    isa     _manual
    cmd     press_key
    key     'J'
""")

lex_decision.productionstring(name="no lexeme found", string="""
    =g>
    isa     goal
    state   'retrieval_done'
    ?retrieval>
    buffer  empty
    state   error
    ==>
    =g>
    isa     goal
    state   'done'
    +manual>
    isa     _manual
    cmd     press_key
    key     'F'
""")

def run_stimulus(word):
    """
    Function running one instance of lexical decision for a word.
    """
    # reset model state to initial state for a new simulation
    # (flush buffers without moving their contents to dec mem)
    try:
        lex_decision.retrieval.pop()
    except KeyError:
        pass
    try:
        lex_decision.goals["g"].pop()
    except KeyError:
        pass
    try:
        lex_decision.goals["imaginal"].pop()
    except KeyError:
        pass

    # reinitialize model
    stim = {1: {'text': word, 'position': (320, 180)}}
    lex_decision.goals["g"].add(actr.makechunk(nameofchunk='start',
                                               typename="goal",
                                               state='attend'))
    lex_decision.goals["imaginal"].add(actr.makechunk(nameofchunk='start',
                                                      typename="word"))
    lex_decision.goals["imaginal"].delay = 0.2
    environment.current_focus = [320,180]
    lex_decision.model_parameters['motor_prepared'] = True #everytime? why not in the beginning?

    # run new simulation; switch to gui=True to suppress pyactr output when estimating Bayesian model
    lex_dec_sim = lex_decision.simulation(realtime=False, gui=False, trace=False,
              environment_process=environment.environment_process,
              stimuli=stim, triggers='', times=10)
    while True:
        lex_dec_sim.step()
        if lex_dec_sim.current_event.action == "KEY PRESSED: J":
            estimated_time = lex_dec_sim.show_time()
            break
        if lex_dec_sim.current_event.action == "KEY PRESSED: F":
            estimated_time = -1
            break
    print(f'Estimated RT for word= {word} is {estimated_time * 1000} ms')
    return estimated_time

def run_lex_decision_task():
    """
    Function running a full lexical decision task:
    it calls run_stimulus(word) for words from all 16 freq bands.
    """
    sample = []
    for word in ORDERED_FREQ:
        sample.append(run_stimulus(word))
    return sample

@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dvector],
       otypes=[pt.dvector])
def actrmodel_latency(lf, le, decay, activation_from_time):
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
    lex_decision.model_parameters["latency_factor"] = lf #np.array(lf).astype("float32").item()
    lex_decision.model_parameters["latency_exponent"] = le #np.array(le).astype("float32").item()
    lex_decision.model_parameters["decay"] = decay #np.array(decay).astype("float32").item()
    activation_dict = {x[0]: x[1] #np.array(x[1]).astype("float32").item()
                       for x in zip(LEMMA_CHUNKS, activation_from_time)}
    lex_decision.decmem.activations.update(activation_dict)
    sample = run_lex_decision_task()
    return np.array(sample)

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
        subvector = scaled_time_vector[(1-compare).nonzero()]
        activation_from_time = pt.log(subvector.sum())
        return activation_from_time
    activation_from_time, _ = pytensor.scan(fn=compute_activation,\
                                          sequences=scaled_time)
    # latency likelihood -- this is where pyactr is used
    pyactr_rt = actrmodel_latency(lf, le, decay, activation_from_time)
    mu_rt = Deterministic('mu_rt', pyactr_rt)
    rt_observed = Normal('rt_observed', mu=mu_rt, sigma=0.01, observed=RT)
    # accuracy likelihood
    odds_reciprocal = pt.exp(-(activation_from_time - threshold)/noise)
    mu_prob = Deterministic('mu_prob', 1/(1 + odds_reciprocal))
    prob_observed = Normal('prob_observed', mu=mu_prob, sigma=0.01,\
                           observed=ACCURACY)
args = args_parser()
with lex_decision_with_bayes:
    num_draws = args.draws #1000
    num_chains = args.chains #4
    num_tunes = args.tunes #10000

    step = pm.DEMetropolisZ(tune="scaling", proposal_dist=pm.NormalProposal)
    trace = pm.sample(draws=num_draws, tune=num_tunes, chains=num_chains, step=step,
                      cores=args.chains)

    print('trace=', trace)
    print('saving trace...')
    trace.to_netcdf(f'{args.root}/book_trace_draws={num_draws}_tune={num_tunes}_chains={num_chains}.nc')
