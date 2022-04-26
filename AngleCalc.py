# import relevant items
import numpy as np
import pandas as pd
import math


# Give a home to Claire's code: [NOT MOST CURRENT]
# The following function takes a dataframe from the downsampling function and calculates angular data from it via Claire's calculation method, now OUTMODED
# this angle calculation does not have a filter
def avt_no_filter(down_sampled_df, frame_start, frame_end):
    # assign down_sampled_df as df
    df = down_sampled_df

    # DATA PROCESSING FOR COORDINATE CONVERTION ###  CLAIRE CALCULATES ANGLE!
    # theta = (0,360) and thetac =(-infinity degrees, infinity degrees)
    # radian to degree conversion
    r2d = 180/np.pi
    # arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)

    df_filter = pd.DataFrame(df.iloc[:, 0])
    # find radius
    df_filter['radius'] = np.power(np.power(
        df['Y displacement (pixels)'], 2) + np.power(df['X displacement (pixels)'], 2), 0.5)
    # find theta arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)
    df_filter['theta'] = - \
        np.arctan2(df['Y displacement (pixels)'],
                   df['X displacement (pixels)'])*r2d
    df_filter['Time (ms)'] = df['Time (ms)']
    # if r is greater than a certain value, the entire row of this dataframe is stored into the next dataframe
    # we conserve the other columns where the row meets the requirement
    df_theta = df_filter.loc[df_filter['radius'] > 0.167].copy()
    # need the .copy() at the end of the line above due to clarity, we want to alter the dataframe to make df_theta, not df_filter
    # arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)
    # add 360 onto the 3rd and 4th quadrant values to make range from (0,360)
    # df_theta is our (0,360) dataframe
    df_theta.loc[df_theta.theta < 0, ['theta']] += 360

    # make dataframe for angular continuous (base dataframe changes with user preferences)
    # df_theta.iloc[:,2] is the (0,360) theta range
    angularc = pd.DataFrame(df_theta.iloc[:, 2])
    angularc.columns = ['theta']
    print('end')
    # add a row of zeros at the top and reset index
    zero_row = pd.DataFrame({'theta': 0}, index=[0])
    angularc = pd.concat([zero_row, angularc]).reset_index(drop=True)

    # find displacement between rows (row[i+1]-row[i]) 350- 25 == 325 --> -35
    angularc['displacement'] = angularc.diff()  # find displcement between rows
    angularc = angularc.apply(pd.to_numeric, errors='coerce')
    angularc = angularc.dropna()  # drop the NANs if there are any
    angularc = angularc.reset_index(drop=True)  # reset the index
    # store the dataframe into an array
    angular_vector = angularc['displacement'].values
    angular_vectorT = angular_vector.T  # transpose the array into a row vector

    # Now we have displacement between rows
    # if the displacement between two rows is greater than 180, subtract 360 (we assume the rotor went backward)
    # if the displacement between two rows is less than -180, add 360 (we assume the rotor went forward)
    # so we edit the displacement to reflect the rotor movement

    angular_vectorT[angular_vectorT >= (180)] -= 360

    angular_vectorT[angular_vectorT <= (-180)] += 360

    # angular_vectorT[sqrt(x**2+(y)**2) < 0.166] = NaN # get this to work
    # df['Y displacement (pixels)']**2 + df['X displacement (pixels)']**2

    # store it back in a pandas dataframe
    disp = angular_vectorT.T
    cont_rotation = pd.DataFrame(
        disp, columns=['theta displacement correction'])

    # add a row of zeros to the top so we conserve the first row
    zero_row = pd.DataFrame({'theta displacement correction': 0}, index=[0])
    cont_rotation = pd.concat([zero_row, cont_rotation]).reset_index(drop=True)
    # enact a culmulitive sum function that adds together all displacements that came before each row
    cont_rotation['continuous theta'] = cont_rotation.cumsum()
    # drop the NAN and or first row of zeros to start at the actual first data point
    cont_rotation = cont_rotation.apply(pd.to_numeric, errors='coerce')

    cont_rotation = cont_rotation.dropna()
    cont_rotation = cont_rotation.reset_index(drop=True)
    cont_rotation.drop(index=cont_rotation.index[0], axis=0, inplace=True)
    # cont_rotation is our (-infinity,infinity) degree rotation dataframe
    cont_rotation = cont_rotation.reset_index(drop=True)
    # Now we have a dataframe called cont_rotation that has 2 columns
    # first column is displacement with the correction and second column is the culmulitive sum of the first col
    # 'continuous theta' is the cumulitive sum of the displacements

    ## Something to look into ##
    # the assumption there is that even though that jump looks like a backwards jump of ~175 degrees, it’s close enough to 180 degrees that the direction could have been mistaken.
    # and if we are unsure if we are mistaken then let’s look at surrounding frames to get a hint for which direction it is going
    # have to do this after calc theta culmulitive

    # MORE OF CLAIRE CLAIRE'S CALCUATION OF ANGLE AND TIME
    # Assign theta(0,360), time, and theta(-infinity,infinity)-->(continuous degree rotation)
    theta = df_theta.iloc[frame_start:frame_end, 2]
    t = df_theta.iloc[frame_start:frame_end, 3]

    thetac = cont_rotation.iloc[frame_start:frame_end, 1]
    return t, theta, thetac


# the following function suppports the most recently updated way to calculate angular orbit data
# this angle calculation goes through user defined radius filtering
# INPUTS:
    # the function must have the inputs of
    # inputs = [down_sampled_df, ind_invalid_reading, rad_filter_type_upper, rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size]
def avt_filter(*inputs):

    # set up block

    # assign inputs
    down_sampled_df, ind_invalid_reading, rad_filter_type_upper, rad_filter_type_lower, z_up, z_down, dist_high, dist_low, bin_size = inputs

    # Assign down_sampled_df as data
    data = down_sampled_df

    #####################################RADIUS FILTERING#####################################
    # labeling my Radius and Zscore data to make it more workable
    my_rad = data["Radius (nm)"]  # Stores Radii into an array
    my_zscore = data["z-score Rad"]  # Stores Z-scores into an array

    # Filter for good data: Upper BOUND
    if rad_filter_type_upper == "zscore":
        up_fil = my_zscore < z_up  # Create a boolean filter for sus zscores that are too high
    else:
        # if we are not using zscore, we use distance. Gather all the upper sus distances
        up_fil = my_rad < dist_high

    # Filter for good data: Lower BOUND
    if rad_filter_type_lower == "zscore":
        # Gather lower sus z scores, if we are talking about zscores.
        down_fil = z_down < my_zscore
    else:
        # Otherwise let's gather the lower distances that are sus.
        down_fil = dist_low < my_rad

    # Filter for valid readings:

    # Section data for Good data (within bounds)
    acceptable_ind = down_fil & up_fil
    # Put Time, X, Y, Radius, and Angle in a dataframe
    data_fil = data.iloc[:, [7, 5, 6, 8, 10]]
    data_fil = data_fil[acceptable_ind]  # Keep only acceptable values

    # Section the eliminated data for both upper and lower
    # Put Time, X, Y, Radius, and Angle in a dataframe
    data_fil_bad = data.iloc[:, [7, 5, 6, 8, 10]]
    # keep only NOT acceptable Values
    data_fil_bad = data_fil_bad[~acceptable_ind]
    # NOTE: Unacceptable values does not include those

    # Section the eliminated data for UPPER and LOWER separately
    # lower:
    data_fil_down_bad = data.iloc[:, [0, 7, 5, 6, 8, 10]]
    # filter for the ones that do not meet the lower cut off
    data_fil_down_bad = data_fil_down_bad[~down_fil]

    # upper
    data_fil_up_bad = data.iloc[:, [0, 7, 5, 6, 8, 10]]
    data_fil_up_bad = data_fil_up_bad[~up_fil]

    # As a reminder , invalid points will be sectioned using
    ind_invalid_reading

    # What data am I graphing? the filtered X values and Filtered Y values
    x_good = data_fil["X displacement (nm)"]
    y_good = data_fil["Y displacement (nm)"]
    x_bad = data_fil_bad["X displacement (nm)"]
    y_bad = data_fil_bad["Y displacement (nm)"]

    ######################################### ANGLE CALCULATION ###########################################

    # Marginal Angle calculation using the my_diff function
    def my_diff(vec):

        vect = vec.diff()  # run a differential on all the angles

        vect[0] = 0  # set the first NaN to 0

        # assuming all increments are less than 180,
        # then make all changes bigger than 180, less than 180.

        # greater than 180 --> negative equivalent
        vect[vect >= (180)] -= 360

        # less than -180 --> positive equivalent
        vect[vect <= (-180)] += 360

        return vect

    # _________________________________________[UNFLITERED DATA]__________________________________________________________

    my_ang_diff = my_diff(data["Angle"])

    # Store corrected differentials into a an array
    data["Delta Angle"] = my_ang_diff

    # Subtract away the orginal angle
    my_ang_cumsum = my_ang_diff.cumsum()

    data["Continuous Angle"] = my_ang_cumsum

    # _________________________________________[FILTERED DATA]__________________________________________________________
    # run the specialized differential function on the Angles and store it
    my_ang_diff = my_diff(data["Angle"])

    # DELTA ANGLE
    # Store corrected differentials into a an array
    data["Delta Angle"] = my_ang_diff

    # CONTINUOUS ANGLE: continuous sumation of the differentials
    # Run running summation fnction on differential angle
    my_ang_cumsum = my_ang_diff.cumsum()

    # Store the cummulative angle data in the "data" dataframe
    data["Continuous Angle"] = my_ang_cumsum

    # Make an array to hold all the data from Angle vs time (avt) data
    avt = data[["index", "Time (ms)", "Angle"]].copy(deep=True)

    avt_good = avt[acceptable_ind]  # Filter for the acceptable indices

    avt_good = pd.DataFrame(avt_good)  # make it its own dataframe

    avt_bad = avt[~acceptable_ind]  # Filter for the unacceptable indices

    avt_bad = pd.DataFrame(avt_bad)  # make it its own dataframe

    # How are we dealing with the BAD Angles?
    # Let's get rid of their values from "data" and replace them as NaN.
    avt_bad["Delta Angle"] = np.nan  # Set all the bad Delta Angles to NaN
    # set all the bad Continuous angles to NaN
    avt_bad["Continuous Angle"] = np.nan

    # How are we dealing with GOOD Angles?
    # Run specialized diff function on the good angles and store into a column in "the good dataframe"
    avt_good["Delta Angle"] = my_diff(avt_good["Angle"])
    # Calculate Delta angle and cmli
    avt_good["Continuous Angle"] = avt_good["Delta Angle"].cumsum()

    # Run Downsampling on filtered Data#####################3
    my_fil_arr = avt_good.to_numpy()  # convert my data frame into numpy array
    # section all columns but the index column
    data_fil_pre_dsa = my_fil_arr[:, 1:5]
    # find the rows of data_fil_pre_dsa
    num_row = np.size(data_fil_pre_dsa, 0)
    # find columns of data_fil_pre_dsa
    num_col = np.size(data_fil_pre_dsa, 1)
    # Re adjust the number of rows with ratio of 1 row per 1 bin_size
    num_row = math.floor(num_row/bin_size)
    # Intialize data table to fill
    data_fil_dsa1 = np.zeros((num_row, num_col))
    vec = np.arange(0, num_col)  # Define variable for for loop

    # This function taken from https://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
    # allows us to downsample by averages over a set number
    # (change 'n' to the number of values you want to average over)
    def average(arr, n):
        end = n * int(len(arr)/n)
        return np.mean(arr[:end].reshape(-1, n), 1)

    for i in vec:  # for each column, run the average downsampling function and store in matrix
        data_fil_dsa1[:, i] = average(data_fil_pre_dsa[:, i], bin_size)
    # make np matrix a dataframe
    data_fil_dsa = pd.DataFrame(data_fil_dsa1, columns=[
                                'Time (ms)', 'Angle', 'Delta Angle', 'Continuous Angle'])

    # packaging values to go
    xy_goodbad = [x_good, y_good, x_bad, y_bad]

    return data, xy_goodbad, avt_good, avt_bad, data_fil_dsa, data_fil_down_bad, data_fil_up_bad
