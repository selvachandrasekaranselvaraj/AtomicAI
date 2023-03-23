import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase.io, os, math, imp, sys
from pytool.tools.analysis.angles import vac_dis_and_angle
from pytool.tools.reader.read_traj import read_traj
#from angles import vac_dis_and_angle

def calssification_plot(number_of_eta, descriptor, rd, ra, classification_type, cluster_size, distance, coord):
    ylabels = ['Cluster numbers', 'dustance from the vacancy', 'coordination number']#, 'tetrahedral angle ($^o$)']
    name_labels = ['cluster', 'distance', 'coordination']#, 'angle']
    n_rd, n_ra = len(rd), len(ra)
    fsize = 17

    for name_label, ylabel in zip(name_labels, ylabels):
        fig, axs = plt.subplots(n_rd, n_ra, figsize=(n_ra*2.5, n_rd*2))#, sharex=True, sharey=True)
        if classification_type == 'lpp':
            class_name = 'LPP'
        elif classification_type == 'pca':
            class_name = 'PCA'
        elif classification_type == 'tsne':
            class_name = 'TsNE'
        elif classification_type == 'tslpp':
            class_name = 'TS-LPP'
      
        st = fig.suptitle(f'{descriptor}_{class_name}', fontsize=fsize*0.9)
        st.set_y(0.96)
        st.set_x(0.4)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
        ylabel_position = int(n_ra * math.floor(n_rd * 0.5) + 1)
        xlabel_position = int(n_ra * (n_rd - 1) + math.ceil(n_ra * 0.5) - 1)
        i, ax = 0, axs.flat
        for d in rd:
            for a in ra:
                if os.path.exists(f'./{classification_type}/{descriptor}_{number_of_eta}_{d}_{a}_{cluster_size}.py'):
                    print(f' Found {descriptor}_{name_label}_{d}_{a}')
                    data = imp.load_source('dummy', f'./{classification_type}/{descriptor}_{number_of_eta}_{d}_{a}_{cluster_size}.py')
                    selected_data_indices = np.array(data.randomly_selected_indices)
                    if name_label == 'cluster':
                       color_values = np.array(data.cluster_labels) + 1
                    if name_label == 'coordination':
                       color_values = np.array(coord)[selected_data_indices]
                    if name_label == 'distance':
                       color_values = np.array(distance)[selected_data_indices]
                    no_of_colors = len(list(set(color_values)))
                    if no_of_colors > 5 :
                       no_of_colors = 6
                    x, y = data.total_xy
                    cmap=plt.cm.get_cmap('jet', no_of_colors) #len(color_interval)),
                    im = ax[i].scatter(
                        x,
                        y,
                        #label=phase,
                        alpha=0.9,
                        edgecolor='none',  #'white',
                        s=6,
                        c=color_values,
                        #linewidth=0.5,
                        cmap=cmap #plt.cm.get_cmap('gnuplot', 12) #len(color_interval)),
                        )
                   #ax[i].set_xlim([min(xmin), max(xmax)])
                   #ax[i].set_ylim([min(ymin), max(ymax)])
                    ax[i].axes.xaxis.set_visible(False)  
                    ax[i].axes.yaxis.set_visible(False)  
                    #if i-1 == 0:
                    #    ax[i - 1].legend(bbox_to_anchor=(0, 1),
                    #                     loc="lower left",
                    #                     #mode="expand",
                    #                     #borderaxespad=0,
                    #                     ncol=3,
                    #                    fontsize=25,
                    #                    frameon=False)
                 
                    if a == min(ra) and d == rd[math.floor(n_rd*0.5)]: #i == ylabel_position:
                        ax[i].set_ylabel('Dimension-2', fontsize=fsize*0.7)
                 
                    if i == xlabel_position:
                        ax[i].set_xlabel('Dimension-1', fontsize=fsize*0.7)
                    ax[i].tick_params(axis='both',
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
                 
                   #ax[i].annotate(r"$\sigma$=" + str(sigma) + '; dm=' + str(dm),
                   #               color="r",
                   #               xy=(0.1, 0.9),
                   #               xycoords='axes fraction',
                   #               fontsize=9,
                   #               ha='left',
                   #               va='top')
                 
                else:
                    print(f' Not Found {descriptor}_{name_label}_{d}_{a}_{cluster_size}')
                    im = ax[i].scatter(None, None)
                    #ax[i].plot(np.arange(-1, 1), np.arange(-1, 1))
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
 
                    if a == min(ra) and d == rd[math.floor(n_rd*0.5)]: #i == ylabel_position:
                        ax[i].set_ylabel('Dimension-2', fontsize=fsize*0.7)
 
                    if i == xlabel_position:
                        ax[i].set_xlabel('Dimension-1', fontsize=fsize*0.7)
                    ax[i].tick_params(axis='both',
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
 
                i += 1
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
 
        work_directory = './images/'
        if not os.path.isdir(work_directory):
            os.makedirs(work_directory)

        cbar = plt.colorbar(im,
                ax=axs,
                label=ylabel,
                #ticks=color_interval, #np.linspace(min(color_values), max(color_values), 10),i
                ticks = np.linspace(min(color_values), max(color_values), no_of_colors),
                pad = 0.1,
                shrink = 0.7
                )
        cbar.set_label(ylabel, size=fsize*0.7)

        for t in cbar.ax.get_yticklabels():
              t.set_fontsize(fsize*0.6)

        descriptor_name = descriptor.split('_')[-1]
        plt.savefig(work_directory+classification_type+'_'+str(cluster_size)+'_'+descriptor_name+'_'+name_label,
                   bbox_inches='tight', 
                   #transparent=True,
                   pad_inches=0.1)
    return 

import sys, os
types_of_classifications = []
for directory in ['lpp', 'pca', 'tsne', 'tslpp']:
    if os.path.isdir(directory):
        types_of_classifications.append(directory)
        print(f'{directory} is availabel here...')
    else:
        print(f'{directory} is not availabel here...')
if len(types_of_classifications) == 0:
    print('No classifiaction directories (lpp, pca, tsne, tslpp) availabel here!!!')
    exit()
#try:
#    classification_type = sys.argv[1]
#except:
#    print()
#    print("Usage: python_exc python_script calssification_type (lpp, tslpp, pca)")
#    print()
#    exit()

number_of_eta = 50 # number of decay functions
descriptor_types = ['ACSF_G2','ACSF_G2G4', 'SOAP'] # 2-body descriptory. Can also be ACSF_G2G4
ra_list = rd_list = np.arange(0.5, 10, 0.1)



rd, ra, des, cluster_nos = [], [], [], []
for classification_type in directories:
    for descriptor_type in descriptor_types:
        for r_d in rd_list: # Descriptor cutoff
            for r_a in ra_list: # Averaging cutoff
                r_d, r_a = round(r_d, 1), round(r_a, 1)
                for cluster_size in [5, 4, 3, 2]:
                    laaf_file = f'./{classification_type}/{descriptor_type}_{number_of_eta}_{r_d}_{r_a}_{cluster_size}.py'
                    if os.path.exists(laaf_file):
                       rd.append(r_d)
                       ra.append(r_a)
                       des.append(descriptor_type)
                       cluster_nos.append(cluster_size)

rd = sorted(set(rd))
ra = sorted(set(ra))
descriptors = sorted(set(des))
cluster_nos = sorted(set(cluster_nos))
print('Rd:', rd,'\nRa:', ra, '\nDescriptors:', descriptors, '\nCluster sizes:', cluster_nos)
#rd = [3.0, 3.5, 4.5, 5.5, 6.5, 7.5]
#ra = [1.0, 3.1]#, 3.5, 4.5, 6.0, 8.0]

if  os.path.exists('trajectory.xyz'):
    traj_file = 'trajectory.xyz'
else:
    print()
    print("trajectory.xyz file does not exist here!!!")
    print()
    exit()

#distance, angles, coord = vac_dis_and_angle(traj_file)
 pca/pca_projected_data.py


for classification_type in types_of_classifications:

    for descriptor in descriptors:
        for cluster_size in cluster_nos: 
    #calssification_plot(number_of_eta, descriptor, rd, ra, distance, distance)
            calssification_plot(number_of_eta, descriptor, rd, ra, classification_type, cluster_size, distance, coord)
