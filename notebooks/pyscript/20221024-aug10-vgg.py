#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import afqinsight.nn.tf_models as nn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from afqinsight.datasets import AFQDataset
from afqinsight.nn.tf_models import cnn_lenet, mlp4, cnn_vgg, lstm1v0, lstm1, lstm2, blstm1, blstm2, lstm_fcn, cnn_resnet
from sklearn.impute import SimpleImputer
import os.path
# Harmonization
from sklearn.model_selection import train_test_split
from neurocombat_sklearn import CombatModel
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

import pickle
import tools
from tools import load_data, model_fit, fit_and_eval, augment_this


# In[3]:


X, y, site = load_data()


# In[ ]:


model_dict = {
  # "cnn_lenet": {"model": cnn_lenet, "lr": 0.001}, 
  # "mlp4": {"model": mlp4, "lr": 0.001},
  "cnn_vgg": {"model": cnn_vgg, "lr": 0.001},
  # "lstm1v0": {"model": lstm1v0, "lr": 0.01},
  # "lstm1": {"model": lstm1, "lr": 0.01},
  # "lstm2": {"model": lstm2, "lr": 0.01},
  # "blstm1": {"model": blstm1, "lr": 0.01},
  # "blstm2": {"model": blstm2, "lr": 0.01},
  # "lstm_fcn": {"model": lstm_fcn, "lr": 0.01},
  # "cnn_resnet": {"model": cnn_resnet, "lr": 0.01}
             }


# In[ ]:


n_runs = 10


# In[ ]:


# scale = 1/5 
# scale = 1/10
# scale = 1/20 
# scale = 1/40


# In[ ]:


# Generated once with this code and then hard-coded:
# seeds = np.array([np.abs(np.floor(np.random.randn()*1000)) for ii in range(n_runs)], dtype=int)

seeds = np.array([484, 645, 714, 244, 215, 1503, 1334, 1576, 469, 1795])


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model, 
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=None,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_1_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_1_pred.csv")


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model,
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=1000,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_2_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_2_pred.csv")


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model, 
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=700,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_3_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_3_pred.csv")


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model, 
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=350,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_4_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_4_pred.csv")


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model,
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=175,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_5_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_5_pred.csv")


# In[ ]:


dfs_eval = []
dfs_pred = []
for model in model_dict:
    print("##################################################")
    print("model: ", model)
    for ii in range(n_runs): 
        print("run: ", ii)
        this_eval, this_pred = fit_and_eval(
            model, 
            model_dict,
            X, 
            y, 
            site,
            random_state=seeds[ii],
            train_size=100,
            augment=augment_this)
        this_eval["run"] = ii
        this_pred["run"] = ii
        dfs_eval.append(this_eval)
        dfs_pred.append(this_pred)
        # Save evaluation metrics
        one_df = pd.concat(dfs_eval)
        one_df.to_csv("vgg_aug10_6_eval.csv")
        # Save predictions and test values:
        one_df = pd.concat(dfs_pred)
        one_df.to_csv("vgg_aug10_6_pred.csv")


# In[ ]:




