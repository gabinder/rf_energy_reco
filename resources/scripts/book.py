#!/usr/bin/env python
"""
Collect random forest features and targets and write to an hdf5 file
"""
from I3Tray import *
from icecube import dataclasses, icetray, VHESelfVeto
from icecube.rf_energy_reco import *
from icecube.rootwriter import I3ROOTWriter
from icecube.hdfwriter import I3HDFWriter
from optparse import OptionParser
import numpy as np
import os
from glob import glob

parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile", default="/data/ana/Muon/level3/sim/2012/neutrino-generator/11069/00000-00999/Level3_nugen_numu_IC86.2012.011069.000???.i3.bz2",
                  dest="infile", help="input file (.i3 format)", type=str)
parser.add_option("-o", "--outfile", default=None, type=str,
                  dest="outfile", help="output file (.i3 format)")
parser.add_option("--hdf5file", default="Level3_nugen_numu_IC86.2012.011069.000XXX.hdf5", type=str,
                  dest="hdf5file", help="output file (.hdf5 format)")
parser.add_option("--rootfile", default=None, type=str,
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

# Insert any cuts here
# Here: numu cc, inside the detector with > 300 m track length, no coincident events
def pre_cut(frame):
    # numu CC only
    wd = frame['I3MCWeightDict']
    if wd['InteractionType']!=1 or abs(wd['PrimaryNeutrinoType'])!=14:
	return False

    # Remove coincident events
    tree = frame['I3MCTree']
    if len(tree.primaries)!=1:
	return False

    # Find in-ice neutrino, muon
    nu = tree.primaries[0]
    if not nu.is_neutrino:
	return False
    child = tree.first_child(nu)
    while child.is_neutrino:
        nu = child
	child = tree.first_child(nu)
    muon = child

    # Find intersection points of muon track with detector volume
    ip = VHESelfVeto.IntersectionsWithInstrumentedVolume(frame['I3Geometry'],muon)
    if len(ip)>0:
        length_before = phys_services.I3Calculator.distance_along_track(muon,ip[0])
        length_after = phys_services.I3Calculator.distance_along_track(muon,ip[-1])
	# Remove if vertex outside detector or contained length < 300 m
	if (length_before > 0) or (length_after < 0):
	    return False
	if (length_after < 300):
	    return False
    else:
	return False

    track = frame['SplineMPE']

    # Remove badly reconstructed events
    if np.degrees(np.arccos(track.dir*muon.dir)) > 5:
        return False

    return True
    
tray.AddModule(pre_cut, 'pre_cut')

# Collect features and insert into frame
tray.AddModule(RandomForestCollect, 'collect',
    TrackName='SplineMPE',
    MillipedeName='SplineMPE_MillipedeHighEnergyMIE',
    NQuantiles=101,
    FeaturesName='RandomForestFeatures',
    TargetName='RandomForestTarget',
    IsStartingTrack=True,
    Cleanup=True)

def simple_weight(frame):
    """
    Astrophysical E^-2 weights
    """
    ow = frame['I3MCWeightDict']['OneWeight']
    e = frame['I3MCWeightDict']['PrimaryNeutrinoEnergy']
    nev = frame['I3MCWeightDict']['NEvents']
    flux = 1e-8*pow(e,-2)
    frame['Weight'] = dataclasses.I3Double(ow*flux/nev)
    return True

tray.AddModule(simple_weight, 'weight')

SubEventStream = opts.stream

if opts.hdf5file:
    tray.AddSegment(I3HDFWriter, 'hdfwriter',
	Output = opts.hdf5file,
	keys = ['I3EventHeader','RandomForestTarget','RandomForestFeatures','Weight'],
        SubEventStreams = [SubEventStream]
    )
if opts.rootfile:
    tray.AddSegment(I3ROOTWriter, 'rootwriter',
	Output = opts.rootfile,
	keys = ['I3EventHeader','RandomForestTarget','RandomForestFeatures','Weight'],
        SubEventStreams = [SubEventStream]
    )
    
if opts.outfile:
    tray.AddModule("I3Writer", "EventWriter",
        FileName = opts.outfile,
        Streams = [icetray.I3Frame.DAQ,
            icetray.I3Frame.Physics],
        DropOrphanStreams=[icetray.I3Frame.DAQ]
    )

tray.Execute()
tray.Finish()

