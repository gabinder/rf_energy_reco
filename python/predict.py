from icecube import dataclasses,icetray,VHESelfVeto,phys_services,simclasses
from sklearn.externals import joblib
from I3Tray import *
import numpy as np

class RandomForestPredict(icetray.I3Module):
    """
    Predict output from an already trained random forest
    """
    def __init__(self,context):
        icetray.I3Module.__init__(self,context)
        self.AddOutBox("OutBox")
        self.AddParameter('FeaturesName', 'Name of random forest features in frame', 'RandomForestFeatures')
        self.AddParameter('OutputName', 'Name of random forest output to put in frame', 'RandomForestOutput')        
        self.AddParameter('RandomForestPickle', 'Location of pickled random forest regressor', 'rf.pkl')

    def Configure(self):
        self.rf = joblib.load(self.GetParameter('RandomForestPickle'))

    def Geometry(self,frame):
        pass

    def Calibration(self,frame):
        pass

    def DetectorStatus(self,frame):
        pass

    def Physics(self,frame):

        features = np.array(frame[self.GetParameter('FeaturesName')])

        # return nan if any feature is nan or inf
        if (np.isnan(features)|np.isinf(features)).sum() > 0:
            output = np.zeros(self.rf.n_outputs_)*np.nan
        else:
            if self.rf.n_outputs_>1:
                output = self.rf.predict(features[None,:])[0]
            else:
                output = self.rf.predict(features[None,:])
        
        frame[self.GetParameter('OutputName')] = dataclasses.I3VectorDouble(output)

        self.PushFrame(frame)
        return True
