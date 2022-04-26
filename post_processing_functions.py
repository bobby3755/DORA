#import packages
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# the following function takes a folder of csv's and turns them into two different
# types of graphs: a histogram of each individual trial at each bins size 
# or the aggregate at each binsize
def csv2Hist(target_folder, name_saving_folder, sample_conditions):
    
    # Change the current directory to target folder
    os.chdir(target_folder)
    # gather all the target files in a list
    target_files = os.listdir()
    
    # Intialize a vector that will become a place holder for all of the lists from the files. 
    dt_bucket = []

    for filename in target_files:
        if '.csv' in filename: 
            dt1 = pd.read_csv(filename, header = None).copy() # create a dataframe from csv file
            dt2 = dt1.loc[1,:].copy() # only take the numeric portion
            dt3 = list(dt2) # convert dt into list
            dt_bucket.append(dt3)
            print("one sum done")

    # the output of dt_bucket is 1 list of n lists (where n = number of csv files/ datasets you have)
    # to aggregate the data you can flatten the lists of lists into one big lists of lists with the below code
    big_bucket = np.hstack(dt_bucket)     
    
    fig, ax = plt.subplots(1, 2) # set up the subplot matrix
    max_dt = max(max(dt_bucket)) # identify the highest value throughout entire list of lists
    min_dt = min(min(dt_bucket)) # identify the lowest value throughout entire list of lists

    # create bins for histogram
    bins = np.linspace(min_dt,max_dt, 10)

    # calculate mean of datasets
    mean0 = sum(big_bucket)/len(big_bucket)

    #labels as mean value
    label0 = "mean = " + str(round(mean0,2))

    #color of line
    color = '#fc4f30'

    # graph vertical line for mean on histograms
    ax[0].axvline(mean0, color = color, label = label0)

    #legend
    ax[0].legend()


    # graph histograms
    ax[0].hist(dt_bucket, bins = bins, edgecolor = "black")

    # Graph labeling 
    ax[0].set_xlabel("Theta in Degrees")
    ax[0].set_ylabel("Number of Jumps")

    #################### SECOND GRAPH

    # graph vertical line for mean on histograms
    ax[1].axvline(mean0, color = color, label = label0)

    #legend
    ax[1].legend()


    # graph histograms
    ax[1].hist(big_bucket, bins = bins, edgecolor = "black")

    # Graph labeling 
    ax[1].set_xlabel("Theta in Degrees")
    ax[1].set_ylabel("Number of Jumps")



    #superior title

    main_title = 'Cumul_'+ sample_conditions + '_Delta_Theta_Distributions_' + 'Clean'

    fig.suptitle(main_title)

    plt.tight_layout()

    

    #building saving information

    # get current path
    path_OG = os.getcwd()

    #make histogram file name
    hist_file_name = main_title + '.png'

    # Create path to the saving folder's path
    my_saving_path = os.path.join(path_OG, hist_file_name)


    plt.savefig(my_saving_path, bbox_inches='tight')

