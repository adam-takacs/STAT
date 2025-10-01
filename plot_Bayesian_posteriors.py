import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src import emulator

# Read in the exp data
with open('data/AllExpData.pkl', 'rb') as handle:
    AllData = pickle.load(handle)
# Read in the emulator
EmulatorPbPb5020 = emulator.Emulator.from_cache('PbPb5020') 
# Read in the parameters, and evaluate the emulator
MCMCSamples = np.load("data/parameter_MCMCSamples_jetRAAandV2.npy")
Examples = MCMCSamples[ np.random.choice(range(len(MCMCSamples)), 2500), :]
Prediction = {"PbPb5020": EmulatorPbPb5020.predict(Examples)}
# Fixed taumin = 0.08 fm
MCMCSamples_early = np.load("data/parameter_MCMCSamples_jetRAAandV2_taumin0.08.npy")
Prediction_early = {"PbPb5020": EmulatorPbPb5020.predict(MCMCSamples_early)}
# Fixed taumin = 0.4 fm
MCMCSamples_late = np.load("data/parameter_MCMCSamples_jetRAAandV2_taumin0.4.npy")
Prediction_late = {"PbPb5020": EmulatorPbPb5020.predict(MCMCSamples_late)}

SystemCount = len(AllData["systems"])
ObsCount = len(AllData['observables'])

with PdfPages('figs/plot_Bayesian_posteriors.pdf') as pdf:
    for i,obs in enumerate(AllData["data"]['PbPb5020']):
        dX_arr = [ [[12, 13, 16, 17, 19, 22, 24, 28, 30, 35, 38, 44, 103, 129, 369], #ATLAS RAA
                    [12,  13,  16,  17,  19,  22,  24,  28,  30,  35,  82, 232],
                    [10, 11, 12, 13, 16, 17, 19, 22, 24, 28, 30, 36, 82, 232],
                    [10, 11, 12, 13, 16, 17, 19, 22, 24, 28, 65, 82, 232],
                    [6, 7, 7, 9, 10, 11, 12, 13, 16, 17, 19, 22, 52, 147]],
                   [[8, 10, 11, 26, 32, 42, 51, 65, 82], # ATLAS v2
                    [8, 10, 11, 26, 32, 42, 51, 65, 82],
                    [8, 10, 11, 26, 32, 42, 51, 65, 82],
                    [8, 10, 11, 26, 32, 42, 51, 65, 82]],
                   [[10, 10, 10, 10, 20, 20, 20], # ALICE RAA
                    [10, 10, 20, 20, 20]],
                   [[100, 100, 500], # CMS RAA
                    [50, 50, 100, 100, 500],
                    [50, 50, 100]],  
                   [[100, 500], # CMS RAA
                    [100, 100, 500],
                    [50, 100]],
                   [[100, 500], # CMS RAA
                    [100, 100, 500],
                    [50, 100]],
                   [[2, 4, 5, 5], # STAR RAA
                    [2, 4, 5, 5],
                    [2, 4, 5, 5]],
                   [[1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60], # ATLAS hadron
                    [1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60],
                    [1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60],
                    [1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60],
                    [1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60],
                    [1.5, 1.7, 1.9, 2.2, 2.7, 3, 3, 4, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60]],
                   [[2.4, 2.4, 4.8, 4.8, 4.8, 6.4, 6.4, 6.4, 12.8, 12.8, 12.8, 17.2, 17.2, 19.2, 25, 85, 150], # CMS hadron
                    [2.4, 2.4, 4.8, 4.8, 4.8, 6.4, 6.4, 6.4, 12.8, 12.8, 12.8, 17.2, 17.2, 19.2, 25, 85, 150],
                    [2.4, 2.4, 4.8, 4.8, 4.8, 6.4, 6.4, 6.4, 12.8, 12.8, 12.8, 17.2, 17.2, 19.2, 25, 85, 150],
                    [2.4, 2.4, 4.8, 4.8, 4.8, 6.4, 6.4, 6.4, 12.8, 12.8, 12.8, 17.2, 17.2, 19.2, 25, 85, 150]],
                   [[3, 7, 10, 20], # ALICE hadron
                    [3, 7, 10, 20],
                    [3, 7, 10, 20],
                    [3, 7, 10, 20],
                    [3, 7, 10, 20],
                    [3, 7, 10, 20]],
                  [[2.5, 4, 4, 5, 5, 5, 5, 10, 10], # ATLAS hadron v2
                   [2.5, 4, 4, 5, 5, 5, 5, 10, 10],
                   [2.5, 4, 4, 5, 5, 5, 5, 10, 10],
                   [2.5, 4, 4, 5, 5, 5, 5, 10, 10],
                   [2.5, 4, 4, 5, 5, 5, 5, 10, 10],
                   [2.5, 4, 4, 5, 5, 5, 5, 10, 10]]
                 ]
        # xlim_arr = [(10,1000), (10,1000), (10,1000), (10,1000), (10,1000), (10,1000), (10,1000), (10,1000)]
        ylim_arr = [(0,1), (-0.03,0.06), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (-0.03,0.1)]
        xylabel_arr = [[r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$v^{\rm jet}_2$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm jet}$ [GeV]', r'$R^{\rm jet}_{AA}$'],
                      [r'$p_T^{\rm h^{\pm}}$ [GeV]', r'$R^{\rm h^{\pm}}_{AA}$'],
                      [r'$p_T^{\rm h^{\pm}}$ [GeV]', r'$R^{\rm h^{\pm}}_{AA}$'],
                      [r'$p_T^{\rm h^{\pm}}$ [GeV]', r'$R^{\rm h^{\pm}}_{AA}$'],
                      [r'$p_T^{\rm h^{\pm}}$ [GeV]', r'$v_2^{\rm h^{\pm}}$']]
        plotlabel_arr = ['ATLAS', 
                         'ATLAS', 
                         'ALICE', 
                         'CMS', 
                         'CMS', 
                         'CMS', 
                         'STAR',
                         'ATLAS',
                         'CMS',
                         'ALICE',
                         'ATLAS']
        title_arr = [[[r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2.8, R_{akt}=0.4$'], 
                      [r'$10-20\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2.8, R_{akt}=0.4$'], 
                      [r'$20-30\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2.8, R_{akt}=0.4$'], 
                      [r'$30-40\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2.8, R_{akt}=0.4$'], 
                      [r'$40-50\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2.8, R_{akt}=0.4$']],
                    [[r'$0-5\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<1.2, R_{akt}=0.2$'],
                     [r'$5-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<1.2, R_{akt}=0.2$'],
                     [r'$10-20\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<1.2, R_{akt}=0.2$'], 
                     [r'$20-40\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<1.2, R_{akt}=0.2$']],
                    [[r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<0.3, R_{akt}=0.2$'],
                     [r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<0.3, R_{akt}=0.4$']],
                    [[r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.2$'],
                     [r'$10-30\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.2$'],
                     [r'$30-50\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.2$']],
                    [[r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.4$'],
                     [r'$10-30\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.4$'],
                     [r'$30-50\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.4$']],
                     [[r'$0-10\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.6$'],
                     [r'$10-30\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.6$'],
                     [r'$30-50\%$ PbPb 5.02 TeV', r'$|\eta^{\rm jet}|<2, R_{akt}=0.6$']],
                    [[r'$0-10\%$ AuAu 200 GeV', r'$|\eta|<1-R_{akt}, R_{akt}=0.2$'],
                     [r'$0-10\%$ AuAu 200 GeV', r'$|\eta|<1-R_{akt}, R_{akt}=0.3$'],
                     [r'$0-10\%$ AuAu 200 GeV', r'$|\eta|<1-R_{akt}, R_{akt}=0.4$']],
                    [[r'$0-5\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'], 
                     [r'$5-10\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'], 
                     [r'$10-20\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'], 
                     [r'$20-30\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'], 
                     [r'$30-40\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$40-50\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$']],
                    [[r'$0-5\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<1$'],
                     [r'$5-10\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<1$'],
                     [r'$10-30\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<1$'],
                     [r'$30-50\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<1$']],
                    [[r'$0-5\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$'],
                     [r'$5-10\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$'],
                     [r'$10-20\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$'],
                     [r'$20-30\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$'],
                     [r'$30-40\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$'],
                     [r'$40-50\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<0.8$']],
                    [[r'$0-5\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$5-10\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$10-20\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$20-30\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$30-40\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                     [r'$40-50\%$ PbPb 5.02 TeV', r'$|\eta^{h^\pm}|<2.5$'],
                    ]]
        titlepos_arr = [[(12,0.1),(12,0.02)],
                        [(12,-0.02),(12,-0.0275)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,0.1),(12,0.02)],
                        [(12,-0.02),(12,-0.0275)],]
        color_line = 'b'
        if i>5: color_line = 'g'
        for j,cent in enumerate(AllData["data"]['PbPb5020'][obs]):
            print(obs,cent)
            figure, ax = plt.subplots(figsize =(5,5))
            # Data
            X = AllData["data"]['PbPb5020'][obs][cent]['x']
            Y = AllData["data"]['PbPb5020'][obs][cent]['y']
            Ystat = AllData["data"]['PbPb5020'][obs][cent]['yerr']['stat'][:,0]
            Ysys = AllData["data"]['PbPb5020'][obs][cent]['yerr']['sys']
            ax.errorbar(X, Y, yerr=Ystat, xerr=np.array(dX_arr[i][j])/2, fmt='ro', label=plotlabel_arr[i])
            ax.bar(X, Ysys.sum(axis=1), bottom=Y-Ysys[:,1], width=dX_arr[i][j], color='r', align='center', alpha=0.3)
            # Predictions:
            for k, y in enumerate(Prediction_early['PbPb5020'][obs][cent]):
                ax.semilogx(X, y, '--', color=color_line, lw=4, alpha=0.1, zorder=100)

            for k, y in enumerate(Prediction['PbPb5020'][obs][cent][::50]):
                ax.semilogx(X, y, '-', color=color_line, lw=4, alpha=0.1, zorder=100)

            for k, y in enumerate(Prediction_late['PbPb5020'][obs][cent]):
                ax.semilogx(X, y, ':', color=color_line, lw=4, alpha=0.1, zorder=100)

            ax.plot(0,0,':', color=color_line, lw=4, label=r'no preeq')
            ax.plot(0,0,'-', color=color_line, lw=4, label=r'best fit')
            ax.plot(0,0,'--', color=color_line, lw=4, label=r'early preeq')

            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=16)
            ax.set_xlabel(xylabel_arr[i][0], fontsize=16)
            ax.set_ylabel(xylabel_arr[i][1], fontsize=16)
            ax.text(*titlepos_arr[i][0],title_arr[i][j][0], fontsize=16)
            ax.text(*titlepos_arr[i][1],title_arr[i][j][1], fontsize=16)
            ax.set_xlim(10,1e3)
            ax.set_ylim(ylim_arr[i])
            ax.legend(fontsize=16)
            pdf.savefig(bbox_inches='tight')  
            plt.close()