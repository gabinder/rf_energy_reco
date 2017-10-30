#!/usr/bin/env python
"""
Plot energy resolution
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize,stats
import tables

h = tables.open_file('Level3_nugen_numu_IC86.2012.011069.00XXXX.hdf5','r')

er = pow(10,h.root.KFoldRandomForestOutput.cols.item[:])

et = pow(10,h.root.RandomForestTarget.cols.item[:])

w = h.root.Weight.cols.value[:]

bins = [np.logspace(2,7,51),np.logspace(2,7,51)]

# joint probability true and reco energy, p(et,er)
# et is indexed rows, er is indexed by columns
p_et_er,bins_et,bins_er = np.histogram2d(et,er,bins=bins,weights=w)
p_et_er /= p_et_er.sum()

plt.figure()
plt.pcolor(bins_et,bins_er,p_et_er.T,norm=plt.matplotlib.colors.LogNorm())
plt.loglog()
plt.xlabel(r'$E_{\nu}^{\mathrm{True}}\,\mathrm{(GeV)}$')
plt.ylabel(r'$E_{\nu}^{\mathrm{Reco}}\,\mathrm{(GeV)}$')
plt.colorbar(label='Probability')

# Now find energy resolution using the method from the energy reco. paper

# probability of er, p(er)
p_et = p_et_er.sum(axis=1)
# probability of et, p(et)
p_er = p_et_er.sum(axis=0)

# conditional probability p(et|er)
p_et_g_er = p_et_er/p_er[None,:]
p_et_g_er[np.isnan(p_et_g_er)] = 0.

# conditional probability p(er|et)
p_er_g_et0 = p_et_er/p_et[:,None]
p_er_g_et0[np.isnan(p_er_g_et0)] = 0.

# conditional probability p(et|et0) = \int p(et|er)p(er|et0) der
# et indexed by rows, et0 indexed by columns
p_et_g_et0 = np.dot(p_et_g_er,p_er_g_et0.T)

# true energy bins centers
cens_et = np.sqrt(bins_et[1:]*bins_et[:-1])

# array to store sigma for each et0
sigma_et = np.zeros_like(cens_et)
for i,et0 in enumerate(cens_et):
    # least-squares gaussian fit for each et0
    def residual(x):
        return np.diff(stats.norm.cdf(np.log10(bins_et) - np.log10(et0),loc=x[0],scale=x[1])) - p_et_g_et0[:,i]
    (mu,sigma),err = optimize.leastsq(residual,[0.0,0.2],ftol=1e-12)
    sigma_et[i] = sigma

plt.figure()
plt.plot(cens_et,sigma_et,lw=2)
plt.xscale('log')
plt.xlabel(r'$E_{\nu}^{\mathrm{True}}\,\mathrm{(GeV)}$') 
plt.ylabel(r'$\sigma_{\log_{10} E_{\nu}}$')   
plt.show()    


