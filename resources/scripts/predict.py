#!/usr/bin/env python
"""
Predict targets using an existing random forest.
"""
from I3Tray import *
from icecube import dataclasses, icetray
from icecube.rf_energy_reco import *
from icecube.rootwriter import I3ROOTWriter
from icecube.hdfwriter import I3HDFWriter
from optparse import OptionParser
import numpy as np
import os
from glob import glob

parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile", default="/data/ana/Muon/level3/sim/2012/neutrino-generator/11069/01000-01999/Level3_nugen_numu_IC86.2012.011069.001000.i3.bz2",
                  dest="infile", help="input file (.i3 format)", type=str)
parser.add_option("-r", "--rf", default="rf.pkl", type=str,
                  dest="rf", help="pickle of random forest")
parser.add_option("-o", "--outfile", default="Level3_nugen_numu_IC86.2012.011069.001000.i3.bz2", type=str,
                  dest="outfile", help="output file (.i3 format)")
parser.add_option("--hdf5file", default="Level3_nugen_numu_IC86.2012.011069.001000.hdf5", type=str,
                  dest="hdf5file", help="output file (.hdf5 format)")
parser.add_option("--rootfile", default="Level3_nugen_numu_IC86.2012.011069.001000.root", type=str,
                  dest="rootfile", help="output file (.root format)")
parser.add_option("-g", "--geofile", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2012.56063_V1.i3.gz", type=str,
                  dest="gcd", help="GCD file")
parser.add_option("-s","--stream", default="Final", type=str,
                  dest="stream", help="stream name")

# parse cmd line args,
(opts,args) = parser.parse_args()

infiles = [opts.gcd] + sorted(glob(opts.infile)) + args

tray = I3Tray()

tray.AddModule('I3Reader', 'reader',
    FilenameList = infiles)

tray.AddModule("Delete",'delete',Keys=['RandomForestFeatures','RandomForestOutput','RandomForestTarget'])

# Collect features and insert into frame
tray.AddModule(RandomForestCollect, 'collect',
    TrackName='SplineMPE',
    MillipedeName='SplineMPE_MillipedeHighEnergyMIE',
    NQuantiles=101,
    FeaturesName='RandomForestFeatures',
    TargetName='RandomForestTarget',
    IsStartingTrack=True,
    Cleanup=False)

# Predict targets and insert into frame
tray.AddModule(RandomForestPredict, 'predict',
    FeaturesName='RandomForestFeatures',
    OutputName='RandomForestOutput',
    RandomForestPickle=opts.rf)

SubEventStream = opts.stream

if opts.hdf5file:
    tray.AddSegment(I3HDFWriter, 'hdfwriter',
                    Output = opts.hdf5file,
	            bookeverything=True,
                    SubEventStreams = [SubEventStream])

if opts.rootfile:
    tray.AddSegment(I3ROOTWriter, 'rootwriter',
                    Output = opts.rootfile,
                    bookeverything=True,
                    SubEventStreams = [SubEventStream])
    
if opts.outfile:
    tray.AddModule("I3Writer", "EventWriter",
                   FileName = opts.outfile,
                   Streams = [icetray.I3Frame.DAQ,
                              icetray.I3Frame.Physics],
                   DropOrphanStreams=[icetray.I3Frame.DAQ])

tray.Execute()
tray.Finish()
