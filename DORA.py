#!/usr/bin/env python
# coding: utf-8

"""
@author: Jerry Wu
Last Update: 4.23.2022
"""
# Note 1: This is the block with all the required imports

import scipy.stats as stats  # added to calculate z-score for Radius filtering
import mplcursors  # Jerry adds way to hover data points in matplotlib rather than plotly
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import ticker
import math
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython import display
import random
import itertools
from matplotlib.widgets import Slider, Button
from IPython.core.display import HTML

# allows for large java HTML
matplotlib.rcParams['animation.embed_limit'] = 2**128


# Note 1: DORA.find_center
# is a code that purely functions to run the centering algoritm on a csv of trajcetories.
# Inputs of this function are a LIST containing the variables:
# file_name, pixel_size, time_step, frame_start, frame_end, cmap, first_zero_end
# Ouptuts are a tuple center with the x and y coordinates of the center.
# Repeatedly run this code until you section of data to work with
'''
Template for the input parameters for DORA.find_center

# Define the below variables in the Variable Bank at the Top
# Relevant variables for this function to be  package into list:

# Initial Parameters is the Relevant Block

file_name = '00021.csv'  # 00086.csv or
pixel_size = 154  # in nanometers
time_step = 20  # miliseconds per frame in trajectory movie
frame_start = 0  # enter 0 to start from beginning of dataset
frame_end = 100  # enter -1 to end at the last value of the data set
cmap = "spring" # enter a color map string from this https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
exp_tag = "Glycerol_50_Laser_50" # a tag that caries the name of the experiment
first_zero_end = 'no'  # yes to cut off all values after first 0,0 = x,y


# packaging all variables into list:
initial_parameters = [file_name,time_step, frame_start, frame_end, cmap, exp_tag,first_zero_end]
center, data, ind_invalid_reading, data_back = find_center(*initial_parameters)
}
'''


def find_center(*relevant_parameters):

    # unpackage our list of variables with their associated variables with variable names
    # unpack list into variables
    file_name, time_step, frame_start, frame_end, cmap, exp_tag, first_zero_end = relevant_parameters

    # I will analyze the raw data from Ryan's code as pre_data and then covert that into two separate parts
    # 1) (data) the data formatted as arrays to be graphed [necessary numbers only]
    # 2) (data_back) the data formated as a Dataframe for record keeping [NaN placed where sus value lies]
    # read the csv file intended
    pre_data = pd.read_csv(file_name, header=None)
    # pre_data = pre_data.dropna()  # drop all NaN values?
    # create an array increasing in steps of
    pre_data['index'] = range(len(pre_data))
    pre_data.columns = ['X position',
                        'Y position', 'index']  # label the columns
    pre_data = pre_data.iloc[:, [2, 0, 1]]  # reorganize the columns
    # create a boolean array of where 1s are when x position is 0 or invalid

    #section data from frame start to frame end
    pre_data = pre_data.iloc[frame_start:frame_end]

    # this is bc Ryan's code exports invalid readings as (0,0)
    ind_invalid_reading = pre_data['X position'] == 0

    if first_zero_end == 'yes':
        # run a boolean through the data set to find 0,0 points
        find_first_0 = (pre_data["X position"] == 0) & (
            pre_data["Y position"] == 0)
        # make an array can pre_x that holds all the x positions
        pre_x = pre_data["X position"].copy()
        # run the boolean through the pre_x variable, all 0,0 are true and stay.
        pre_x = pre_data[find_first_0]
        # the first 0,0 is the first item in this new pre_x
        my_first_0 = pre_x.index[0]
        frame_end = my_first_0  # set the index of the first 0,0 point as the end frame

    # SEPARATE data into front and back end (front==graphing ; back == tables)
    # if the index is not invalid (or valid) keep it and store in data
    data = pre_data[~ind_invalid_reading].copy()
    # section the pre data for all the invalid values
    data_back = pre_data[ind_invalid_reading].copy()

    # in data back develop a time colomn
    data_back['Time (ms)'] = data_back['index']*time_step

    # set all target x positions to NaN, if the reading was sus
    data_back['X position'] = np.nan
    # set all target y positions to NaN, if the reading was sus
    data_back['Y position'] = np.nan
    data_back['Sus Type'] = 'Invalid Reading'

    ####################################### CENTERING ALGORITHM ###############################################

    # establish empty lists
    stand = []

    # find uniform guesses in range of max and min unaltered data values for y position
    # THE NUMBER OF UNIFORM GUESSES IS CURRENTLY HARD CODED AT 50 FOR X AND Y, CULMULITIVE 2,500
    guess_y = np.linspace(data.iloc[:, 2].max(
    ), data.iloc[:, 2].min(), 50)
    # put into list
    guess_y = guess_y.tolist()
    # find guesses for x position
    guess_x = np.linspace(data.iloc[:, 1].max(
    ), data.iloc[:, 1].min(), 50)
    guess_x = guess_x.tolist()

    # permute each x and y center guess together to create 10,000 unique center guesses
    center_guesses = list(itertools.product(guess_x, guess_y))
    # store center guesses in dataframe
    c = pd.DataFrame(center_guesses, columns=['X', 'Y'])
    # set up list to store average distances (radius) of circular trajectory path
    ave_distance = []
    # set up list to store standard deviation of distances to each point in the trajectory
    stand = []
    j = 0
    for j in range(len(c)):  # chnage to range(len(c))
        # find the distance between each point in a dataframe against guess[i]
        distance = np.power(
            ((data["X position"] - c['X'][j])**2 + (data["Y position"] - c['Y'][j])**2), 0.5)
        # store distances in a dataframe
        d = pd.DataFrame(distance, columns=['distance'])
        # find average of distances (this would be the radius)
        ave_d = d['distance'].mean(axis=0)
        # store all average distances from each guess[i] distance dataframes into list
        ave_distance.append(ave_d)
        # find standard deviation of center distance from each point in trajectory for each guess[i]
        std = d['distance'].std(axis=0)
        # store each standard deviation in a list
        stand.append(std)

        j += 1
    # put radius and std lists in a dataframe
    c['average_distance'] = ave_distance
    c['std'] = stand

    # this block finds the row with the lowest std, the corresponding radius and x,y coordinates for the center
    # want to return row with lowest std
    target_row = c['std'].idxmin()

    # x center guess with lowest std
    center_x = c.loc[target_row, 'X']
    # y center guess with lowest std
    center_y = c.loc[target_row, 'Y']
    # radius of trajectory
    dist = c.loc[target_row, 'average_distance']

    ###########GRAPHING BLOCK

    # Our regularly scheduled 2D graphing program
    fig = plt.figure(figsize=(6, 6), dpi=100)
    # 121 # 1X1 grid plot 1, subplot(222) would be 2X2 grid plot 2, (223)--> 2X2 plot 3
    ax = fig.add_subplot(111)

    # Set up for color bar
    #HARD CODE:
    z_axis_label = "Frames" 

    #collect x and y values
    x =  data.iloc[:, 1] 
    y = data.iloc[:, 2]

    # A color bar associated with time needs two things c and cmap
    #these arguments go into ax.scatter as args

    # c (A scalar or sequence of n numbers to be mapped to colors using cmap and norm.)
    c = data["index"]

    #Make a ticks vector that spans the total number of frames
    if frame_end == -1:
        last_frame = len(pre_data.iloc[:,0])
    else:
        last_frame = frame_end
    tix = np.linspace(frame_start,last_frame,8)
    tix_1 = np.round(tix,0)


    #scatter plot with a color vector
    p = ax.scatter(x, y, c=c, cmap = cmap,alpha=0.7)
    #add a vertical side bar that defines the color
    plt.colorbar(p, label=z_axis_label, shrink=.82, ticks = tix_1 )


    plt.axis('square')
    plt.xticks(rotation=45)

    # add a red dot to indicate center of trajectory
    ax.scatter(center_x, center_y, color='red')
    plt.text(x=center_x + 0.02, y=center_y + 0.02, s='algorithm centering')

    # add a red dot to indicate center of trajectory
    ax.scatter(center_x, center_y, color='red')
    plt.text(x=center_x + 0.02, y=center_y + 0.02, s='algorithm centering')

    # add a circle with center at our best guess and radius derived from our best guess
    circle = plt.Circle((center_x, center_y), dist, color='r', fill=False)
    ax.add_patch(circle)

    # Colorbar parameters below if we want one in the future

    # cbar = plt.colorbar(p, label= 'time' ,                asdfshrink= .82) #

    # #setting the ticks on the colorbar to span the length of the time column with 6 increments
    # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # tix = np.linspace(0,len(data),6, dtype = int) # forces colorbar to show time in integers
    # tix_c = tix*20
    # cbar.set_ticklabels(tix_c)

    plt.axis('square')  # INTEGRAL to maintaining aspect ratio
    plt.xticks(rotation=45)
    ax.set_xlabel('X position (unaltered)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Y position (unaltered)', fontweight='bold', fontsize=14)

    # plot title and font configurations

    # take the file name and separate from the extension
    # the first value in the tuple is the number
    # the second is .csv 
    # the number 00086.csv is the peak --> so this code takes the peak number
    pk = os.path.splitext(file_name)[0]

    graph_type = 'Algorithm_Center_Guess'

    # change title order!!! 
    list_of_strings = [graph_type,exp_tag, pk ]

    #in quotes is the the delimiter between the items in the string
    # by default it is a _ 
    my_title = "_".join(list_of_strings)

    plt.title(my_title, fontweight='bold', fontsize=16)
    plt.show()

    # #below is the key to maintaining the value of the center variable once the user satisfaction is achieved
    # global center

    center = (center_x, center_y)

    print('The center is {0}'.format(center))

    print('If the center is satisfactory, change the find_center_coordinates parameter to no')
    print('If the center is unsatisfactory, adjust the frame_start and frame_end parameters and try again')

    return center, data, ind_invalid_reading, data_back


# Note 2: DORA.downsample
# is a code that runs a down sampling method on the data
# it takes the following inputs:
# bin_size, processing, data, center, time_step, pixel_size, frame_start, frame_end
#      MOST are self explanatory, but for reference
#           "processing" is the type of processing to be run: 1)
#           "data" is the data from DORA.find_center


'''
# Relevant USER Input Parameters from the Variable Bank

bin_size = 20  # bin size for downsample/filter processing
processing = "none"  # enter downsample, moving average, or none

# DORA.down_sample(*downsample_parameters)
downsample_parameters = [bin_size, processing, data, center, time_step, pixel_size, frame_start, frame_end]
down_sampled_df = DORA.downsample(*downsample_parameters)

'''


def downsample(*downsample_parameters):
    # To hand our list input, we need to unpack it and redefine the values with the respective variable names
    bin_size, processing, data, center, time_step, pixel_size, frame_start, frame_end = downsample_parameters

    ## DATA NORMILIZATION AND UNIT ASSIGNMENT ##

    # # find the average of X and Y column respectively
    # ave = data.mean(axis=0) --> Jerry thinks this is antiquated code

    # substract averages from each column to find displacement, store into new columns
    data["X displacement (pixels)"] = data['X position'] - center[0]
    data["Y displacement (pixels)"] = data['Y position'] - center[1]
    # mutiply pixel displacement columns by scaler to find nm displacement, store in new columns
    data["X displacement (nm)"] = data['X displacement (pixels)']*pixel_size
    data["Y displacement (nm)"] = data['Y displacement (pixels)']*pixel_size
    # multiply the index counter column by time_step to make a time step column, store into new column
    data["Time (ms)"] = data['index']*time_step

    # drop all NaN values *not a number --> Jerry does not think this is necessary any more
    # data = data.dropna()
    # #drop NAN try to conserve time (what if we have NAN in x and not in Y? need to drop the whole row)

    # Recalculation of center using distance forumla -- Jerry
    # Radius Calculation from distance formula
    data['Radius (nm)'] = np.power(((data["X displacement (nm)"])
                                    ** 2 + (data["Y displacement (nm)"])**2), 0.5)

    # Z score calculation
    data['z-score Rad'] = stats.zscore(data["Radius (nm)"])

    # Angle Calculation

    # Radian to degree conversion factor
    r2d = 180/np.pi

    # Take Arc Tan function of x and y coord to get radius. Arctan 2 makes Quad 3 and 4 negative.
    data['Angle'] = -np.arctan2(data['Y displacement (nm)'],
                                data['X displacement (nm)'])*r2d

    # Make all negative Theta values positive equivalents
    data.loc[data.Angle < 0, ['Angle']] += 360

    ######################### PROCESSING BLOCK ##############################

    # Simple Moving Average or "filter" dataframe:
    ma = pd.DataFrame(data.iloc[:, 0], columns=['index'])

    window = bin_size
    # Built in simple moving average function is applied to normal data and stored in dataframe "ma"
    ma['X movement'] = data.iloc[:, 1].rolling(window=window).mean()
    ma['Y movement'] = data.iloc[:, 2].rolling(window=window).mean()
    ma['X displacement (pixels)'] = data.iloc[:,
                                              3].rolling(window=window).mean()
    ma['Y displacement (pixels)'] = data.iloc[:,
                                              4].rolling(window=window).mean()
    ma['X displacement (nm)'] = data.iloc[:, 5].rolling(window=window).mean()
    ma['Y displacement (nm)'] = data.iloc[:, 6].rolling(window=window).mean()
    ma['Time (ms)'] = data.iloc[:, 7].rolling(window=window).mean()

    # This block delets the null spaces in the new dataframe and realigns the data
    ma = ma.apply(pd.to_numeric, errors='coerce')
    ma = ma.dropna()
    ma = ma.reset_index(drop=True)

    # Downsampling dataframe:
    da = pd.DataFrame(data.iloc[:, :])
    # divide original index by sample size and round to nearest whole number to
    # achieve new index number underwhich the origial index is stored
    u = math.floor(frame_start/bin_size)
    v = math.floor(frame_end/bin_size)

    # isolate the column (if we print this it will show as a dataframe with 2 cols: indexes and time values)
    daT_column = da.iloc[:, 7]
    daDY_column = da.iloc[:, 6]
    daDX_column = da.iloc[:, 5]
    daPY_column = da.iloc[:, 4]
    daPX_column = da.iloc[:, 3]
    daI_column = da.iloc[:, 0]
    daX_column = da.iloc[:, 1]
    daY_column = da.iloc[:, 2]
    # We just want the values in the column
    daT = daT_column.values
    daDY = daDY_column.values
    daDX = daDX_column.values
    daPY = daPY_column.values
    daPX = daPX_column.values
    daI = daI_column.values
    daX = daX_column.values
    daY = daY_column.values
    # This function taken from https://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
    # allows us to downsample by averages over a set number
    # (change 'n' to the number of values you want to average over)

    def average(arr, n):
        end = n * int(len(arr)/n)
        return np.mean(arr[:end].reshape(-1, n), 1)
    # Takes a column from our 'da' dataframe and runs the function over it
    # stores the new values in variables as an array (values in a row)

    # assigning each new row to a varialble
    Time = average(daT, bin_size)
    Index = average(daI, bin_size)
    Xda = average(daX, bin_size)
    Yda = average(daY, bin_size)
    Ydisnm = average(daDY, bin_size)
    Xdisnm = average(daDX, bin_size)
    YdisP = average(daPY, bin_size)
    XdisP = average(daPX, bin_size)

    # reshaping the data in a 1D column
    TimeT = Time[:, np.newaxis]
    YdisnmT = Ydisnm[:, np.newaxis]
    XdisnmT = Xdisnm[:, np.newaxis]
    YdisPT = YdisP[:, np.newaxis]
    XdisPT = XdisP[:, np.newaxis]
    XdaT = Xda[:, np.newaxis]
    YdaT = Yda[:, np.newaxis]
    IndexT = Index[:, np.newaxis]

    # stores in a new dataframe 'dsa' for: downsampling average
    dsa = pd.DataFrame(IndexT, columns=['index'])
    # appending to our data frame
    dsa['X movement'] = XdaT
    dsa['Y movement'] = YdaT
    dsa['X displacement (pixels)'] = XdisPT
    dsa['Y displacement (pixels)'] = YdisPT
    dsa['X displacement (nm)'] = XdisnmT
    dsa['Y displacement (nm)'] = YdisnmT
    dsa['Time (ms)'] = TimeT

    # DETERMINE PROCESSING AND UNIT TYPE:
    # if more processing methods are to be added, an if statement must be
    # added with a key word to select that data frame
    # "df" becomes the variable used in the graphing block below
    if processing == "none":
        df = data
        return df
    if processing == "moving average":
        df = ma
        return df
    if processing == "downsample":
        df = dsa
        frame_start = math.floor(frame_start/bin_size)
        frame_end = math.floor(frame_end/bin_size)
        return df, frame_start, frame_end


# Note 3: DORA.graph(plot_type,*graph_parameters)
# This function graphs the following plots:
# Graphing options:
    # Trajectory Maps:
    # 2D: Colorful 2D visulization of the rotor from above
    # 3D: 2D plot but time is an axis

    # Grid plot
    # grid: a grid of little snippets of the data

    # Angular Analysis:

    #         By Jerry
    # radius_filter: Demarcate the sus data points that will be eliminated from calculations
    # find_sus_angle: Indicate sus angles within angular_continuous by Jerry
    # angular_continuous_filtered: Angular Continuous recalculated with sus points filtered. Sus skips indicated.
    # basal3: Graphs tailored for the basal graph analysis 3/14/2022
    # Angular Continuous with a downsampled curve as well. still has bugs with error labelling

    #         By Claire:  [NOT DONE]
    # angular: angle vs time, but it's not cummulative and resets at 360 to 0 (Claire)
    # angular_continuous: Claire's Calculation of a cummulative angle
    # find_sus_angle_CR: Indicate sus angles within angular_continuous by Claire's calculations

    # Animation   [NOT DONE]
    # interactive: Interactive graph
    # animated: animated trajectory in notebook
    # HTML: Animated trajectory in a new window. May run better
'''
Template for DORA.graph(*INPUTS)

use the following for graph_parameters

#Trajectory map parameters:
tajectory_map_parameters = [file_name, down_sampled_df, plot_type, display_center, title, x_axis_label, y_axis_label, z_axis_label, unit, 
pixel_min, pixel_max, axis_increment_nm, axis_increment_pixel, nm_min, nm_max, save_plot, frame_start, frame_end, time_step,cmap,exp_tag]

#Angle Versus Time (AVT or avt) parameters:
avt_parameters = [file_name, down_sampled_df, plot_type, display_center, ind_invalid_reading, rad_filter_type_upper,
                  rad_filter_type_lower, z_up, z_down, dist_high, dist_low, graph_style, bin_size, frame_start, frame_end,
                  display_center, title, x_axis_label, y_axis_label, z_axis_label, unit, pixel_min, pixel_max,
                  axis_increment_nm, axis_increment_pixel, nm_min, nm_max, save_plot, data_back, cmap,exp_tag] 
'''


def graph(plot_type, *graph_parameters):

    ###################### Which graph will be graphed?##########

    # graph groupings:

    # create a list of the acceptable groupings for the trajectory maps
    trajectory_map = ["2D", "3D"]

    # create a list of the acceptable groupings for the Angle Time grouping
    AngleTime = ["radius_filter", "find_sus_angle", "angular_continuous_filtered",
                 "basal3", "angular", "angular_continuous", "find_sus_angle_CR"]

    if plot_type in trajectory_map:

        #### Set up block#####

        # Accept my variables from graphing parameters
        [file_name, down_sampled_df, plot_type, display_center, title, x_axis_label, y_axis_label, z_axis_label, unit, pixel_min, pixel_max,
            axis_increment_nm, axis_increment_pixel, nm_min, nm_max, save_plot, frame_start, frame_end, time_step, cmap, exp_tag] = graph_parameters

        # Claire's code accepts down_sampled_df as df
        df = down_sampled_df

        #####################Graphing data assignment block##############
        # Here the code determines the units of the graph, only for cartesian graphs
        if unit == "pixel":
            x_unit = 3
            y_unit = 4
        if unit == "nm":
            x_unit = 5
            y_unit = 6
        # assign values of x y and z
        # move this outside this block to apply for all "none"
        x = df.iloc[frame_start:frame_end, x_unit]
        y = df.iloc[frame_start:frame_end, y_unit]
        z = df.iloc[frame_start:frame_end, 7]  # col 7 is the time col

        # graph either
        if plot_type == "2D":

            # Let the graphing beghin!!!
            fig = plt.figure(figsize=(6, 6), dpi=100)
            # this comand is here to take advantage of the "axes" plotting library
            ax = fig.add_subplot(111)

            # Set up for color bar
            #HARD CODE:
            z_axis_label = "Frames" 

            # A color bar associated with time needs two things c and cmap
            #these arguments go into ax.scatter as args

            # c (A scalar or sequence of n numbers to be mapped to colors using cmap and norm.)
            c = df["index"]

            #Make a ticks vector that spans the total number of frames
            tix = np.linspace(frame_start,frame_end,8)
            tix_1 = np.round(tix,0)


            #scatter plot with a color vector
            p = ax.scatter(x, y, c=c, cmap = cmap,alpha=0.7)
            #add a vertical side bar that defines the color
            plt.colorbar(p, label=z_axis_label, shrink=.82, ticks = tix_1 )


            plt.axis('square')
            plt.xticks(rotation=45)

            # display center
            if display_center == "yes":
                # in a centered graph, the center is actually(0,0)
                center1 = [0, 0]
                # plots center point as magenta X
                ax.scatter(0, 0, color='Magenta', marker="X", s=150)
                plt.text(x=center1[0] + 0.02,
                         y=center1[1] + 0.02, s='CENTER')

            # set graph limit conditions depending on unit specified
            if unit == "pixel":
                ax.set_xlim(pixel_min, pixel_max)
                ax.set_ylim(pixel_min, pixel_max)
                ax.yaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_pixel))
                ax.xaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_pixel))
            if unit == "nm":
                ax.set_xlim(nm_min, nm_max)
                ax.set_ylim(nm_min, nm_max)
                ax.yaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_nm))
                ax.xaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_nm))

            # Jerry Adds a hover cursor
            mplcursors.cursor(hover=True)
            mplcursors.cursor(highlight=True)

            # axis labels and font configurations
            ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
            ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)

            # plot title and font configurations

            # take the file name and separate from the extension
            # the first value in the tuple is the number
            # the second is .csv 
            # the number 00086.csv is the peak --> so this code takes the peak number
            pk = os.path.splitext(file_name)[0]

            graph_type = '2D_Map'

            # change title order!!! 
            list_of_strings = [graph_type, exp_tag, pk]

            #in quotes is the the delimiter between the items in the string
            # by default it is a _ 
            my_title = "_".join(list_of_strings)

            plt.title(my_title, fontweight='bold', fontsize=16)

            if save_plot == "yes":
                # put title input and date time
                plt.savefig(title+"_2D.png")
        if plot_type == "3D":
            # This block splices the segments between data points and assigns each segment to a color
            points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segs, cmap=plt.get_cmap('cool'))
            lc.set_array(z)

            # This block plots the figure at a specified size, in 3D configuration, sets axis range, gathers the
            # colored segments from above, and assigns labels
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca(projection='3d')
            ax.set_zlim(min(z), max(z))

            # define graphing proportions according to unit used
            if unit == "pixel":
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            if unit == "nm":
                ax.set_xlim(-150, 150)
                ax.set_ylim(-150, 150)

            ax.add_collection3d(lc, zs=z, zdir='z')

            # define title
            
            # take the file name and separate from the extension
            # the first value in the tuple is the number
            # the second is .csv 
            # the number 00086.csv is the peak --> so this code takes the peak number
            pk = os.path.splitext(file_name)[0]

            graph_type = '2D_Map'

            # change title order!!! 
            list_of_strings = [graph_type, exp_tag, pk]

            #in quotes is the the delimiter between the items in the string
            # by default it is a _ 
            my_title = "_".join(list_of_strings)

            plt.title(my_title, fontweight='bold', fontsize=16)

            # set labels
            ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
            ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)
            ax.set_zlabel(z_axis_label, fontweight='bold', fontsize=14)

            if save_plot == 'yes':
                plt.savefig(title+'_3D.png', dpi=300)

            plt.show()

    if plot_type in AngleTime:

        # Set up Block

        # Accept my variables from graphing parameters
        (file_name, down_sampled_df, plot_type, display_center, ind_invalid_reading, rad_filter_type_upper,
         rad_filter_type_lower, z_up, z_down, dist_high, dist_low, graph_style, bin_size, frame_start, frame_end,
         display_center, title, x_axis_label, y_axis_label, z_axis_label, unit, pixel_min, pixel_max,
         axis_increment_nm, axis_increment_pixel, nm_min, nm_max, save_plot, data_back, cmap, exp_tag) = graph_parameters

        # import the 2 ways to analyze angular data: 1) Claire's way, 2) Jerry's Way [More Current]
        import AngleCalc

        # Gather inputs to Cacluate Angle under Jerry's Angle calculation paradigm
        inputs_avt_filter = [down_sampled_df, ind_invalid_reading, rad_filter_type_upper,
                             rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size]
        # Call and Run Jerry's Angle Calculation
        data, xy_goodbad, avt_good, avt_bad, data_fil_dsa, data_fil_down_bad, data_fil_up_bad = AngleCalc.avt_filter(
            *inputs_avt_filter)

        # Claire's code accepts down_sampled_df as df
        df = down_sampled_df

        #####################Graphing data assignment block##############
        # Here the code determines the units of the graph, only for cartesian graphs
        if unit == "pixel":
            x_unit = 3
            y_unit = 4
        if unit == "nm":
            x_unit = 5
            y_unit = 6
        # assign values of x y and z
        # move this outside this block to apply for all "none"
        x = df.iloc[frame_start:frame_end, x_unit]
        y = df.iloc[frame_start:frame_end, y_unit]
        z = df.iloc[frame_start:frame_end, 7]  # col 7 is the time col
        
        # unpack list xy_goodbad into respective outputs
        x_good, y_good, x_bad, y_bad = xy_goodbad

        # Calculate The angle according to Claire's method
        t, theta, thetac = AngleCalc.avt_no_filter(
            down_sampled_df, frame_start, frame_end)

        # Let the graphing begin:

        if plot_type == "radius_filter":
            fig = plt.figure(figsize=(7, 6), dpi=100)
            # this comand is here to take advantage of the "axes" plotting library
            ax = fig.add_subplot(111)

            # Set up for color bar
            #HARD CODE:
            z_axis_label = "Frames" 

            # A color bar associated with time needs two things c and cmap
            #these arguments go into ax.scatter as args

            # c (A scalar or sequence of n numbers to be mapped to colors using cmap and norm.)
            c = df["index"]

            #Make a ticks vector that spans the total number of frames
            tix = np.linspace(frame_start,frame_end,8)
            tix_1 = np.round(tix,0)

            #scatter plot with a color vector
            p = ax.scatter(x, y, c=c, cmap = cmap,alpha=0.7)
            #add a vertical side bar that defines the color
            plt.colorbar(p, label=z_axis_label, shrink=.82, ticks = tix_1 )


            plt.axis('square')
            plt.xticks(rotation=45)

            # set graph limit conditions depending on unit specified
            if unit == "pixel":
                ax.set_xlim(pixel_min, pixel_max)
                ax.set_ylim(pixel_min, pixel_max)
                ax.yaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_pixel))
                ax.xaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_pixel))
            if unit == "nm":
                ax.set_xlim(nm_min, nm_max)
                ax.set_ylim(nm_min, nm_max)
                ax.yaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_nm))
                ax.xaxis.set_major_locator(
                    ticker.LinearLocator(axis_increment_nm))

            # display center
            if display_center == "yes":
                # in a centered graph, the center is actually(0,0)
                center1 = [0, 0]
                # plots center point as magenta X
                ax.scatter(0, 0, color='Magenta', marker="X", s=150)
                plt.text(x=center1[0] + 0.02,
                         y=center1[1] + 0.02, s='CENTER')

            # Jerry Adds a hover cursor
            mplcursors.cursor(hover=True)
            mplcursors.cursor(highlight=True)

            # axis labels and font configurations
            ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
            ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)

            # plot title and font configurations
            # take the file name and separate from the extension
            # the first value in the tuple is the number
            # the second is .csv 
            # the number 00086.csv is the peak --> so this code takes the peak number
            pk = os.path.splitext(file_name)[0]

            graph_type = plot_type

            # change title order!!! 
            list_of_strings = [graph_type, exp_tag, pk]

            #in quotes is the the delimiter between the items in the string
            # by default it is a _ 
            my_title = "_".join(list_of_strings)
            plt.title(my_title, fontweight='bold', fontsize=16)

            if save_plot == "yes":
                plt.savefig(title+"_2D.png")  # put title input and date time
        if plot_type == "find_sus_angle":

            # data organization
            times = data["Time (ms)"]
            conti_angle = data["Continuous Angle"]

            # setup fig and ax
            fig, ax = plt.subplots()

            # choose scatter plot or line plot
            if graph_style == 'scatter':
                ax.scatter(times, conti_angle, c='b', s=5)
            else:
                line, = plt.plot(times, conti_angle, 'b')

            # setting up for making vertical lines or indications of bad points (high and low points)
            ang_min = min(conti_angle)
            ang_max = max(conti_angle)

            # find and graph lower bad values
            bad_times_down = data_fil_down_bad["Time (ms)"]
            ax.vlines([bad_times_down], ang_min, ang_max,
                      linestyles='dashed', colors='maroon')

            # find and graph upper bad values
            bad_times_up = data_fil_up_bad["Time (ms)"]
            ax.vlines([bad_times_up], ang_min, ang_max,
                      linestyles='dashed', colors='tomato')

            # find and graph the invalid bad values
            invalid_times = data_back["Time (ms)"]
            ax.vlines([invalid_times], ang_min, ang_max,
                      linestyles='dashed', colors='darkcyan')

            # Legend
            ax.legend(['Angle data', 'Lower Sus', 'Upper Sus',
                      'Invalid Readings'], loc='lower right')

            # formatting
            plt.rcParams['figure.figsize'] = [13, 6]

            plt.xlabel('Time (ms)')
            plt.ylabel('Angle Accumulation (degrees)')

            # Title configuration

            # take the file name and separate from the extension
            # the first value in the tuple is the number
            # the second is .csv 
            # the number 00086.csv is the peak --> so this code takes the peak number
            pk = os.path.splitext(file_name)[0]

            graph_type = 'Accumulation of Angle (degrees) as a function of Time (ms)'

            # change title order!!! 
            list_of_strings = [exp_tag, pk]

            #in quotes is the the delimiter between the items in the string
            # by default it is a _ 
            my_title = "_".join(list_of_strings)
            my_title = graph_type + "\n" + my_title #add a line break between graph name and defining parameters
            plt.title(my_title)

            plt.xlim(0, max(times)+1000)
#             plt.ylim(-180,max(conti_angle)+180)
            ax.set_xticks(np.arange(0, max(times)+1000, 1000))
            ax.set_yticks(np.arange(-360, max(conti_angle)+180, 180))
            plt.xticks(rotation=-45)
            plt.grid()

            # hovering attempt 2
            # added by Jerry for Matplotlib compatible hovering
            mplcursors.cursor(hover=True)
            # Graph the newly calcuated Angular Continuous data, now filtered for good points only
        if plot_type == "angular_continuous_filtered":

            # data organization
            times = avt_good["Time (ms)"]
            conti_angle = avt_good["Continuous Angle"]

            # Graph a Scatter Plot otherwize the hover tool hovers to made up points
            fig, ax = plt.subplots()

            if graph_style == 'scatter':
                ax.scatter(times, conti_angle, c='b', s=5)
            else:
                line, = plt.plot(times, conti_angle, 'b')

            # setting up for making vertical lines or indications of bad points (high and low points)
            ang_min = min(conti_angle)
            ang_max = max(conti_angle)

            # find and graph lower bad values
            bad_times_down = data_fil_down_bad["Time (ms)"]
            ax.vlines([bad_times_down], ang_min, ang_max,
                      linestyles='dashed', colors='red')

            # find and graph upper bad values
            bad_times_up = data_fil_up_bad["Time (ms)"]
            ax.vlines([bad_times_up], ang_min, ang_max,
                      linestyles='dashed', colors='black')

            # find and graph the invalid bad values
            invalid_times = data_back["Time (ms)"]
            ax.vlines([invalid_times], ang_min, ang_max,
                      linestyles='dashed', colors='darkcyan')

            # Legend
            ax.legend(['Angle data', 'Lower Sus', 'Upper Sus',
                      'Invalid Readings'], loc='lower right')

            # formatting
            plt.rcParams['figure.figsize'] = [20, 6]

            plt.xlabel('Time (ms)')
            plt.ylabel('Angle Accumulation (degrees)')

            # Title configuration

            # take the file name and separate from the extension
            # the first value in the tuple is the number
            # the second is .csv 
            # the number 00086.csv is the peak --> so this code takes the peak number
            pk = os.path.splitext(file_name)[0]

            graph_type = 'Accumulation of Angle (degrees) as a function of Time (ms)'

            # change title order!!! 
            list_of_strings = [exp_tag, pk]

            #in quotes is the the delimiter between the items in the string
            # by default it is a _ 
            my_title = "_".join(list_of_strings)
            my_title = graph_type + "\n" + my_title #add a line break between graph name and defining parameters
            plt.title(my_title)

            plt.xlim(0, max(times)+1000)
            plt.ylim(ang_min-180, ang_max+180)
            ax.set_xticks(np.arange(0, max(times)+1000, 1000))
            ax.set_yticks(np.arange(ang_min-180, ang_max+180, 180))
            plt.xticks(rotation=-45)
            plt.grid()

            # hovering attempt 2
            # added by Jerry for Matplotlib compatible hovering
            mplcursors.cursor(hover=True)


# Note 4: DORA.table
# the following code creates the relevant data tables from the inputs detailed below in the template section
# the outputs are 4 data tables and, if you so choose, a saved .csv of a data table

'''
Template for DataTable inputs

table_parameters = [down_sampled_df, ind_invalid_reading, rad_filter_type_upper,
                    rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size, data_back,save_table,file_name]
                    
'''


def table(*table_parameters):
    # Set up Block

    # Accept my variables from graphing parameters
    (down_sampled_df, ind_invalid_reading, rad_filter_type_upper,
     rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size, data_back, save_table, file_name) = table_parameters

    # import the 2 ways to analyze angular data: 1) Claire's way, 2) Jerry's Way [More Current]
    import AngleCalc

    # Gather inputs to Cacluate Angle under Jerry's Angle calculation paradigm
    inputs_avt_filter = [down_sampled_df, ind_invalid_reading, rad_filter_type_upper,
                         rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size]
    # Call and Run Jerry's Angle Calculation
    data, xy_goodbad, avt_good, avt_bad, data_fil_dsa, data_fil_down_bad, data_fil_up_bad = AngleCalc.avt_filter(
        *inputs_avt_filter)

    ################################### [Final Data Table Assembly ] ######################################

    # Organzize Data Table with Final Filtered Data [Re insert sus points from lower and upper bound filtering]
    # slap all the bad data on the end of the good data
    data_final = pd.concat([avt_good, avt_bad])
    # sort by index so that values go back to where they are supposed to be:
    data_final = data_final.sort_values(by=["index"])

    # re insert sus points [re insert sus points from invalid]
    # Organzize Data Table with Front End data (data) and back end data (data_back)
    # slap all the bad data on the end of the good data
    data_final_final = pd.concat([data_back, data_final])
    # sort by index so that values go back to where they are supposed to be:
    data_final_final = data_final_final.sort_values(by=["index"])

    del data_final_final['X position']
    del data_final_final['Y position']

    # Label of the Data with either Normal, Upper bound, Lower Bound, Invalid Reading

    # Initialize the data table to be populated
    data_final_final["Sus Type"] = 'None'
    # store this into a dummy vector
    dummy_vec = data_final_final["Sus Type"].copy()

    # Set all indices of AVT_good to normal
    # Select all indicies that are NORMAL --> avt_good indicies
    ind_Normal = avt_good["index"].copy()
    dummy_vec[ind_Normal] = 'Normal'

    # Set all indicies of data_back to 'Invalid Reading' and put them in the dummy variable
    ind_IR = data_back["index"].copy()
    dummy_vec[ind_IR] = 'Invalid Reading'

    # Find bad lower bounds and index for them and set value to "below bound"
    ind_bad_down = data_fil_down_bad["index"].copy()
    dummy_vec[ind_bad_down] = 'Below Bound'

    # fFind bad upper bounds and index for them and set value to "upper bound"
    ind_bad_up = data_fil_up_bad["index"].copy()
    dummy_vec[ind_bad_up] = 'Above Bound'

    data_final_final["Sus Type"] = dummy_vec
    data_final_final = data_final_final[[
        "index", "Time (ms)", "Angle", "Delta Angle", "Continuous Angle", "Sus Type"]]

    if save_table == "yes":
        my_title = file_name + "_Final_DataTable.csv"
        data_final_final.to_csv(
            my_title, index="false")
        print("I have saved the table for " + file_name + " as " + my_title)

    return data, avt_good, avt_bad, data_final_final

# extract the relevant data from dataframe and convert into lists


def collect_variable(DataTable, col, file_name, sample_conditions, name_saving_folder):
    # this function operates on the final data exported by DORA.table
    # it takes 1) the data table 2) the column you want to operate on
    # and it gives you that column as a csv
    # 3) is the filename that this data came from
    # 4) are the name of the experimental set up so that you know which data this csv comes from
    # 5) name of the folder you want to save your outputted column .csv's into

    # Intialize my lists to store values
    collection_pot = []

    # Extract the Target Variable or col varible from both
    var_col = list(DataTable[col].copy())

    # Take extracted data and save into a my collection pot list
    collection_pot.append(var_col)

    # convert the extracted data into a dataframe
    df_collection_pot = pd.DataFrame(collection_pot)

    # set the name of the file to be saved
    my_file_name = sample_conditions + col + "_from_" + file_name + ".csv"

    # create a folder to save CSV into called "name_saving_folder"

    # if the folder does not exist already
    if not os.path.exists(name_saving_folder):
        # make a folder in the current path
        os.mkdir(name_saving_folder)
        # and report this to the user
        print("Directory ", name_saving_folder,  " Created ")
    else:
        # else tell user it exists already
        print("Directory ", name_saving_folder,  " already exists")

    # Where are we trying to save the file?
    # ANSWER: Current directory/name_saving_folder/my_file_name.csv

    # get current path
    path_OG = os.getcwd()

    # Create path to the saving folder's path
    my_saving_path = os.path.join(path_OG, name_saving_folder, my_file_name)

    # save the dataframe
    df_collection_pot.to_csv(my_saving_path)

    print('I have SAVED the file, {FileName}, in the directory {FolderName}'.format(
        FileName="my_file_name", FolderName="name_saving_folder"))
