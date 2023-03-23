import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io, os, math, imp, sys
#from pytool.tools.analysis.angles import vac_dis_and_angle
#from pytool.tools.reader.read_traj import read_traj
#from angles import vac_dis_and_angle

def calssification_plot():
    classification_types = []
    for directory in ['lpp', 'pca', 'tsne', 'tslpp']:
        if os.path.isdir(directory):
            classification_types.append(directory)
            print(f'{directory} is availabel here...')
        else:
            print(f'{directory} is not availabel here...')
    if not os.path.isdir('structure_analysis'):
            print('structure_analysis directory is availabel here...')
            exit()
    manual_classification = imp.load_source('dummy', './structure_analysis/structure_analysis.py').structure_analysis['std']
    if len(classification_types) == 0:
        print('No classifiaction directories (lpp, pca, tsne, tslpp) availabel here!!!')
        exit()

    availabel_descriptors = ['ACSF_G2','ACSF_G2G4', 'SOAP'] # 2-body descriptory. Can also be ACSF_G2G4
    ylabel = name_label = 'std'
    data = imp.load_source('dummy', './pca/pca_projected_data.py')
    keys = list(data.pca_projected.keys())
    descriptors, dr, ar = [], [], []  # descriptors type, descriptors radius, avarage radius
    descriptors, dr, ar, sy1, sy2 = np.array([key.split('_') for key in keys]).transpose()
    descriptors = list(set(descriptors))
    rd = sorted(list(set(dr)))
    ra = sorted(list(set(ar)))
    n_rd, n_ra = len(rd), len(ra)
    fsize = 17
    for classification_type in classification_types:
        for descriptor in descriptors:
         
            if classification_type == 'lpp':
                classification_name = 'LPP'
            elif classification_type == 'pca':
                classification_name = 'PCA'
            elif classification_type == 'tsne':
                classification_name = 'TsNE'
            elif classification_type == 'tslpp':
                classification_name = 'TS-LPP'
         
            fig, axs = plt.subplots(n_rd, n_ra, figsize=(n_ra*2.5, n_rd*2))#, sharex=True, sharey=True)
            fig.suptitle(f'{descriptor}_{classification_name}', x= 0.4, y=0.96, fontsize=fsize*0.9) #subtitle
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
            ylabel_position = int(n_ra * math.floor(n_rd * 0.5) + 1)
            xlabel_position = int(n_ra * (n_rd - 1) + math.ceil(n_ra * 0.5) - 1)
            plot_i, ax = 0, axs.flat
            for d in rd:
                for a in ra:
                    laaf_key = f'{descriptor}_{d}_{a}_Si_Si'
                    xy = data.pca_projected[laaf_key]
                    x, y = xy[0], xy[1]
                    cmap=plt.cm.get_cmap('jet')#, no_of_colors) #len(color_interval)),
                    im = ax[plot_i].scatter(
                        x,
                        y,
                        #label=phase,
                        alpha=0.9,
                        edgecolor='none',  #'white',
                        s=6,
                        c=manual_classification, #color_values,
                        #linewidth=0.5,
                        cmap=cmap #plt.cm.get_cmap('gnuplot', 12) #len(color_interval)),
                        )
                  # ax[plot_i].set_xlim([min(x), max(x)])
                  # ax[plot_i].set_ylim([min(y), max(y)])                        

                    if a == min(ra) and d == rd[math.floor(n_rd*0.5)]: #i == ylabel_position:
                        ax[plot_i].set_ylabel('Dimension-2', fontsize=fsize*0.7)
                    
                    if plot_i == xlabel_position:
                        ax[plot_i].set_xlabel('Dimension-1', fontsize=fsize*0.7)
                    ax[plot_i].tick_params(axis='both',
                                   which='both',
                                   direction='in',
                                   length=0,
                                   width=4,
                                   left=True,
                                   top=True,
                                   right=True,
                                   labelleft=True,
                                   labelsize=fsize*0.2,
                                   pad=0)
                    ax[plot_i].axes.xaxis.set_visible(False)
                    ax[plot_i].axes.yaxis.set_visible(False)
                    plot_i += 1
            cols = ['$R_a$={}$\AA$'.format(col) for col in ra]
            rows = ['$R_d$={}$\AA$'.format(row) for row in rd]
            for ax, col in zip(axs[0], cols):
                ax.set_title(col, fontsize=fsize*0.6)
         
            for ax in axs[:, 0]:
                ax.axes.yaxis.set_visible(True)
         
            for ax in axs[len(rd)-1]:
                ax.axes.xaxis.set_visible(True)
         
            for ax, row in zip(axs[:, len(ra)-1], rows):
                #ax.set_ylabel(row, rotation=0, fontsize=25)
                ax.annotate(row, xy=(1, 0.5), xycoords='axes fraction', fontsize=fsize*0.6,
                        xytext=(-0, 0), textcoords='offset points',
                        ha='left', va='center')
         
            color_values = manual_classification 
            cbar = plt.colorbar(im,
                    ax=axs,
                    label=None, #ylabel,
                    ticks=np.linspace(min(color_values), max(color_values), 2),
                    #ticks = np.linspace(min(color_values), max(color_values)),
                    pad = 0.1,
                    shrink = 0.7
                    )
            #cbar.set_label(ylabel, size=fsize*0.7)
            cbar.set_ticklabels(['Nondefect', 'Defect'], size=fsize*0.7)  
            #for t in cbar.ax.get_yticklabels():
            #      t.set_fontsize(fsize*0.6)
         
            work_directory = './images/'
            if not os.path.isdir(work_directory):
                os.makedirs(work_directory)
            plt.savefig(work_directory+classification_type+'_'+descriptor,
                       bbox_inches='tight',
                       #transparent=True,
                       pad_inches=0.1)



    return
calssification_plot()

