#!/usr/bin/env python
"""
Script to train a random forest with optional k-fold cross-validation
"""
import numpy as np
import matplotlib.pyplot as plt
import tables
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_predict
from glob import glob
import os
from optparse import OptionParser

parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-j", "--jobs", default=1, type=int,
                  dest="jobs", help="number of jobs to run in random forest")
parser.add_option("-i", "--infile", default="Level3_nugen_numu_IC86.2012.011069.000XXX.hdf5", type=str,
                  dest="infile", help="input file(s) (.hdf5 format) containing feature, target, and weight arrays")
parser.add_option("-o", "--outfile", default="rf.pkl", type=str,
                  dest="outfile", help="output pickle of random forest (.pkl)")
parser.add_option("-m", "--merged", default=1000, type=int,
                  dest="merged", help="number of files merged into one")
parser.add_option("-w", "--write", action="store_true", default=False,
                  dest="write", help="flag to write predictions into input hdf5 file")
parser.add_option("-k", "--kfolds", default=3, type=int,
                  dest="kfolds", help="number of folds for k-fold cross validation")

# parse cmd line args,
(opts,args) = parser.parse_args()

if opts.write:
    h = tables.open_file(opts.infile,'r+')
else:
    h = tables.open_file(opts.infile,'r')

# load event weights
sample_weight = h.root.Weight.cols.value[:]/opts.merged

# load features array, shape = (n_events,n_features)
x = h.root.RandomForestFeatures.cols.item[:]
n_features = h.root.RandomForestFeatures.cols.vector_index[:].max() + 1
x = x.reshape((len(sample_weight),n_features))

# load target array, shape = (n_events,n_targets) if n_targets>1.
y = h.root.RandomForestTarget.cols.item[:]
n_targets = h.root.RandomForestTarget.cols.vector_index[:].max()+1
if n_targets > 1:
    y = y.reshape((len(sample_weight),n_targets))

# create random forest and specify your additional options here
rf = RandomForestRegressor(n_estimators=400,n_jobs=opts.jobs)

if opts.kfolds > 1:
    # cross validated predictions
    y_pred=cross_val_predict(rf,x,y,fit_params=dict(sample_weight=sample_weight),cv=opts.kfolds,n_jobs=opts.jobs)

    if opts.write:
        # Write cross-validated predictions to hdf5 file
        predict = np.copy(h.root.RandomForestTarget.cols[:])
        predict['item'] = y_pred.flatten()
        if hasattr(h.root,'KFoldRandomForestOutput'):
            h.root.KFoldRandomForestOutput.cols[:] = predict
        else:
            h.create_table(h.root,'KFoldRandomForestOutput',predict)

# Train with full sample and save pickle
rf.fit(x,y,sample_weight=sample_weight)
joblib.dump(rf,opts.outfile)

h.close()
