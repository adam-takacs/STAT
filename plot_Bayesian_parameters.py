import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

Allfit_noV2_MCMCSamples = np.load("data/parameter_MCMCSamples_jetRAA.npy")
Allfit_MCMCSamples = np.load("data/parameter_MCMCSamples_jetRAAandV2.npy")

NDimension = 2
Ranges = np.array([[0,1.2],[0.4,1.6]])
figure, axes = plt.subplots(figsize = (3 * NDimension, 3 * NDimension), ncols = NDimension, nrows = NDimension)
Names = [r'$\tau_{\rm min}$', r'$g_{\rm med}$']
yLabels = [r'$p(\tau_{\rm min})$', r'$p(g_{\rm med})$']
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
        if i==j:
            ax.hist(Allfit_noV2_MCMCSamples[:,i], bins=50,
                    range=Ranges[:,i], histtype='step', lw=2, color='C0', density=True, label=r'only $R_{AA}$')
            ax.hist(Allfit_MCMCSamples[:,i], bins=50,
                    range=Ranges[:,i], histtype='step', lw=2, color='C3', density=True, label=r'$R_{AA}$ & $v_2$')
            ax.set_xlabel(Names[i], fontsize=14)
            ax.set_xlim(*Ranges[:,j])
            if i==0: 
                ax.set_ylim(0,16)
                ax.legend(loc=(0.01,0.7),frameon=False,fontsize=12)
        if i>j:
            ax.hist2d(Allfit_MCMCSamples[:, j], Allfit_MCMCSamples[:, i], 
                      bins=50, range=[Ranges[:,j], Ranges[:,i]], 
                      cmap='Reds')
            ax.set_xlabel(Names[j], fontsize=14)
            ax.set_ylabel(Names[i], fontsize=14)
            ax.set_xlim(*Ranges[:,j])
            ax.set_ylim(*Ranges[:,i])
        if i<j:
            ax.hist2d(Allfit_noV2_MCMCSamples[:, i], Allfit_noV2_MCMCSamples[:, j], 
                      bins=50, range=[Ranges[:,i], Ranges[:,j]], 
                      cmap='Blues')
            ax.set_xlabel(Names[i], fontsize=14)
            ax.set_ylabel(Names[j], fontsize=14)
            ax.set_xlim(*Ranges[:,i])
            ax.set_ylim(*Ranges[:,j])
plt.tight_layout()
plt.savefig('figs/plot_Bayesian_parameters.pdf', dpi = 192)
