# -*- coding: utf-8 -*-
"""
CCA Assessment - Development
"""

import subprocess
import sys

subprocess.call(["pip","install","-r", "./requirements.txt"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from girth import twopl_mml
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import irt
import xarray as xr

df = pd.read_csv(sys.argv[1],sep=";",decimal=",").fillna(0)
data = np.array(df[df.columns[7:]].transpose()).astype('int')
df[sys.argv[2]] = df[sys.argv[2]].str.rstrip('%').astype('float')/100

# Solve for parameters
estimates = twopl_mml(data)

# Unpack estimates
discrimination_estimates = estimates['Discrimination']
difficulty_estimates = estimates['Difficulty']

probPctg = pd.DataFrame(pd.DataFrame(data).sum(axis=1)*100/pd.DataFrame(data).count(axis=1)).transpose()
longProbPctg = pd.melt(probPctg)

#PLOTTING DISTRIBUTION AND HEATMAP OF QUESTION DIFFICULTY
fig, axes = plt.subplots(2,1,figsize=(16,12))
sns.kdeplot(df[sys.argv[2]],color="purple", fill=True,ax=axes[0])
axes[0].set_xlim(0.0, 1.0)
axes[0].axvline(df[sys.argv[2]].mean(),linestyle="--",color='purple')
axes[0].axvline(df[sys.argv[2]].median(),linestyle="-.",color='red')
axes[0].set_title("Score Distribution",fontweight="bold",fontsize="15")
axes[0].set(xlabel=None)

sns.barplot(y=longProbPctg.value,x=longProbPctg.variable,color="purple", fill=True,ax=axes[1])
axes[1].xaxis.set_tick_params(labelsize=10,rotation=90)
axes[1].set_title("Question Difficulty",fontweight="bold",fontsize="15")
axes[1].set(xlabel=None)
plt.savefig('./results/distribution.pdf')

corrMatrix = pd.DataFrame(data.transpose()).corr().fillna(0)
fig = plt.subplots(figsize=(10,10))
sns.heatmap(corrMatrix, annot=False,center=0)
plt.savefig('./results/heatmap.pdf')


#EXPORTING POSTERIOR DISTRIBUTION FOR PEOPLE ABILITY AND QUESTION DIFFICULTIES
with pm.Model() as model:
    ## Independent priors
    alpha = pm.Normal('Person', mu = 0, sigma = 3, shape = (1, len(data.transpose())))
    gamma = pm.Normal('Question', mu = 0, sigma = 3, shape = (data.transpose().shape[1], 1))

    ## Log-Likelihood
    def logp(d):
        v1 = tt.transpose(d) * tt.log(tt.nnet.sigmoid(alpha - (gamma - gamma.mean(0))))
        v2 = tt.transpose((1-d)) * tt.log(1 - tt.nnet.sigmoid(alpha - (gamma - gamma.mean(0))))
        return v1 + v2

    ll = pm.DensityDist('ll', logp, observed = {'d': data.transpose()})
    trace = pm.sample(2000, cores=-1, step = pm.NUTS(),return_inferencedata=False)
    trace = trace[250:]

    xr.set_options(display_style="text")
    rng = np.random.default_rng()

    fig, axes = plt.subplots(ncols=2,nrows=len(df)+1,figsize=(16,16))
    az.rcParams["plot.max_subplots"] = 200
    az.plot_trace(trace, var_names="Person",kind="trace",plot_kwargs={'lw':2},
                  chain_prop={"ls":"-"},combined=True,compact=False,
                  axes=axes,fill_kwargs={'color':'orange','alpha': 0.5},
                  backend_kwargs={"Figure.title_location":"left"}
                 )
    plt.xlim([-5,5])
    fig.savefig('./results/distPerson.pdf')

    fig, axes = plt.subplots(ncols=2,nrows=len(data)+1,figsize=(16,16))
    ax = az.plot_trace(trace, var_names="Question", chain_prop={"ls":"-"},plot_kwargs={'lw':2,'textsize':7},combined=True,compact=False,
                axes=axes,fill_kwargs={"color":"orange",'alpha': 0.8})
    plt.xlim([-5,10])
    fig.savefig('./results/distQuestion.pdf')

estimates3=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in estimates.items() ]))[['Discrimination','Difficulty','Ability']]
estimates3.to_csv("./results/estimates.csv")
