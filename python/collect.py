from icecube import dataclasses,icetray,VHESelfVeto,phys_services,simclasses
from sklearn.externals import joblib
from I3Tray import *
import numpy as np

class RandomForestCollect(icetray.I3Module):
    """
    Collect features and targets for random forest training
    """
    def __init__(self,context):
        icetray.I3Module.__init__(self,context)
        self.AddOutBox("OutBox")
        self.AddParameter('TrackName', 'Name of reco. track', 'SplineMPE')
        self.AddParameter('MillipedeName', 'Name of millipede loss vector', 'SplineMPE_MillipedeHighEnergyMIE')
        self.AddParameter('FeaturesName', 'Name of output feature vector', 'RandomForestFeatures')
        self.AddParameter('TargetName', 'Name of target vector', 'RandomForestTarget')
        self.AddParameter('NQuantiles','Number of quantiles of energy loss distribution to use', 101)
        self.AddParameter('IsStartingTrack', 'Learn neutrino energy/inelasticity of starting tracks? Otherwise learn most energetic muon energy.', True)
        self.AddParameter('DoInelasticity','Learn hadronic and muon energies of starting tracks', False)
        self.AddParameter('Cleanup', 'Remove features/targets that are nan/inf', True)

    def Configure(self):
        self.trackName = self.GetParameter('TrackName')
        self.nQuantiles = self.GetParameter('NQuantiles')
        self.isStartingTrack = self.GetParameter('IsStartingTrack')
        self.doInelasticity = self.GetParameter('DoInelasticity')

    def Geometry(self,frame):
        self.PushFrame(frame)
        return True

    def Calibration(self,frame):
        pass

    def DetectorStatus(self,frame):
        pass

    def Physics(self,frame):
        track = frame[self.trackName]

        # Find detector entrance and exit times
        ip = VHESelfVeto.IntersectionsWithInstrumentedVolume(frame['I3Geometry'],track)
        if len(ip)>0:
            t_start = phys_services.I3Calculator.distance_along_track(track,ip[0])/track.speed + track.time
            t_stop = phys_services.I3Calculator.distance_along_track(track,ip[-1])/track.speed + track.time
        else:
            t_start = float('nan')
            t_stop = float('nan')

        losses = frame[self.GetParameter('MillipedeName')]
        
        # Exclude losses outside detector
        dist = []
        energy = []
        for loss in losses:
            if (loss.time > t_start) and (loss.time < t_stop):
                dist.append((loss.time - t_start)*track.speed)
                energy.append(loss.energy)
        dist = np.array(dist)
        energy = np.array(energy)

        # Length of track in detector
        length = (t_stop - t_start)*track.speed

        # Compute quantiles of energy loss distribution
        if len(dist)>0:
            quantiles = length - np.interp(np.linspace(0,1,self.nQuantiles),np.cumsum(energy)/np.sum(energy),dist)        
        else:
            quantiles = np.zeros(self.nQuantiles)*np.nan

        # Make features used for training [log10(contained energy),contained track length, 0th quantile, 1st quantile, ...]
        features = [np.log10(np.sum(energy)),length]+quantiles.tolist()

        # Cleanup nans and infs since scikit-learn can't handle this
        if self.GetParameter('Cleanup'):
            if (np.isnan(features)|np.isinf(features)).sum() > 0:
                return False
        
        frame[self.GetParameter('FeaturesName')] = dataclasses.I3VectorDouble(features)

        # Add target values to frame if this is mc
        # target is in-ice neutrino energy for starting tracks
        if self.isStartingTrack:
            nu_inice_energy = 0.
            if frame.Has('I3MCTree'):
                # Find in-ice neutrino
                tree = frame['I3MCTree']
                nu = tree.primaries[0]
                child = tree.first_child(nu)
                while child.is_neutrino:
                    if tree.number_of_children(child)>0:                        
                        nu = child
                        child = tree.first_child(nu)
                    else:
                        break
                nu_inice_energy = nu.energy

                # get muon energy and cascade energy if learning inelasticity
                if self.doInelasticity:
                    muon_energy = 0.
                    if abs(child.type)==13:
                        muon_energy = child.energy
                    had_energy = nu_inice_energy - muon_energy
                    targets = [np.log10(had_energy),np.log10(muon_energy)]
                else:
                    targets = [np.log10(nu_inice_energy)]

        # otherwise target is most enegetic muon energy at point of closest approach to detector center
        else:
            muon_center_energy = 0.
            if frame.Has('MMCTrackList'):
                for mmc_track in frame['MMCTrackList']:
                    if mmc_track.Ec > muon_center_energy:
                        muon_center_energy = mmc_track.Ec
            targets = [np.log10(muon_center_energy)]

        # Cleanup nans and infs
        if self.GetParameter('Cleanup'):
            if (np.isnan(targets)|np.isinf(targets)).sum() > 0:
                return False

        frame[self.GetParameter('TargetName')] = dataclasses.I3VectorDouble(targets)

        self.PushFrame(frame)
        return True
