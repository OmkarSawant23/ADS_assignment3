#Import statements
import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors as err

def read_csv_file(filename):
    '''
    This function reads the csv files and also filters the dataframe \
    and returns two dataframe one with transposed data and one with
    non transposed data.

    ----------
    filename : string 
        Returns the pandas dataframe.
    -------
    year_index : pandas dataframe.
        This is a pandas dataframe which returns years as index.
    country_index : pandas dataframe.
        This is a pandas dataframe which returns years as index.

    '''

    # set year range and country list to filter the dataset
    start_from_yeart = 1995
    end_to_year = 2020
   

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # clean na data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    country_index = wb_df.loc[:,all_cols_list]

    # make the country as index and then drop column as it becomes index
    country_index.index = country_index["Country Name"]
    country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    country_index.columns = country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    year_index = pd.DataFrame.transpose(country_index)

    # return the two dataframes year as index and country as index
    return year_index, country_index

def clustering_data(fert_data_cw, cr_lnd_cw, ncluster, year_1, year_2):
    '''
    This function plots the graph with help of clustering algorithms.
    ----------
    fert_data_cw : pandas dataframe
        returns the pandas dataframe with specifed year.
    cr_lnd_cw : pandas dataframe
        returns the pandas dataframe with specifed year.
    ncluster : string
        returns number of clusters.
    year_1 : string
        returns the specifed year.
    year_2 : string
        returnsthe specifed year.

    
    ------
    None.

    '''
    #saving the dataframe
    ar_cluster_1 = pd.DataFrame()
    #selecting year
    ar_cluster_1["fert"] = fert_data_cw.loc[:, year_1]
    ar_cluster_1["arable"] = cr_lnd_cw.loc[:, year_1]
    ar_cluster_1["fert_2000"] = fert_data_cw.loc[:, year_2]
    ar_cluster_1["cereal_2000"] = cr_lnd_cw.loc[:, year_2]
    
    #dropping non zero values.
    ar_cluster_1 = ar_cluster_1.dropna()
    
    #selecting clusters.
    df_norm, df_min, df_max = ct.scaler(ar_cluster_1)
    
    #determining best nymber of clusters.
    for ic in range(2, 11):
        score = one_silhoutte(ar_cluster_1, ic)
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
        
    #machine learning code  
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    kmeans.fit(df_norm)

    labels = kmeans.labels_
    
    #cluster centers.
    cen = kmeans.cluster_centers_
    cen = ct.backscale(cen, df_min, df_max)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    
    #variable for simplicity of code.
    x1 = ar_cluster_1["fert"]
    y1 = ar_cluster_1["arable"]
    x2 = ar_cluster_1["fert_2000"]
    y2 = ar_cluster_1["cereal_2000"]
    
    #for doing subplots.
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.colormaps["Paired"]
    fig, c_f = plt.subplots(1, 2, figsize=(12, 5)) 
    
    #First scatter subplot.
    scatter1 = c_f[0].scatter(x1, y1, 10, labels, marker="o", cmap=cm)
    c_f[0].scatter(xkmeans, ykmeans, 45, "k", marker="d")
    #setting labels, title, limits and legend.
    c_f[0].set_xlabel("Fertilizer(kg)")
    c_f[0].set_ylabel("Cereal")
    c_f[0].set_title(f"Cereal yield vs fertilizer usage ({year_1})")
    c_f[0].set_ylim(0, 30000)
    c_f[0].grid(True)
    hand1, labels1 = scatter1.legend_elements()
    c_f[0].legend(hand1, ["Cluster 1", "Cluster 2"])


    #second scatter subplot.
    scatter2 = c_f[1].scatter(x2, y2, s=10, c=labels, marker="o", cmap=cm)
    c_f[1].scatter(xkmeans, ykmeans, 45, "k", marker="d")
    #setting labels, title, limits and legend.
    c_f[1].set_xlabel("Fertilizer(kg)")
    c_f[1].set_ylabel("Cereal")
    c_f[1].set_title(f"Cereal yield vs fertilizer usage ({year_2})")
    c_f[1].set_ylim(0, 30000)
    c_f[1].grid(True)
    hand2, labels1 = scatter1.legend_elements()
    c_f[1].legend(hand2, ["Cluster 1", "Cluster 2"])
    #saving image as png.
    plt.savefig("cluster.png", dpi = 300)

    plt.show()
    
    
def fit_data(fert_data_yw, ar_lnd_yw, country) :
    '''
    This function implements fitting algorithm to data and plots fitting graph.
    ----------
    fert_data_yw :  pandas dataframe
        returns the pandas dataframe with specifed country.
    ar_lnd_yw :  pandas dataframe
        returns the pandas dataframe with specifed country.
    country : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #saving the dataframe
    ar_cluster = pd.DataFrame()
    #selecting specific country.
    ar_cluster["fert_k"] = fert_data_yw[country]
    ar_cluster["arable_k"] = ar_lnd_yw[country]
    #dropping non zero values.
    ar_cluster = ar_cluster.dropna()
    #plotting the figure.
    plt.figure()
    #setting year as index.
    ar_cluster["Year"] = ar_cluster.index

    #parameter and covariance matrix. 
    param, covar = opt.curve_fit(poly, ar_cluster["Year"], ar_cluster["fert_k"])
    ar_cluster["fit"] = poly(ar_cluster["Year"], *param)
    
    #first fitting plot
    ar_cluster.plot("Year", ["fert_k", "fit"])
    #setting title, labels.
    plt.title(f"{country} fertilizer consumption(kg per hectare of arable land)")
    plt.xlabel("Year")
    plt.ylabel("Fertilizer consumption(kg)")
    #saving figure.
    plt.savefig("fit1.png", dpi = 300)

     #parameter and covariance matrix. 
    param_1, covar_1 = opt.curve_fit(poly, ar_cluster["Year"], ar_cluster["arable_k"])
    ar_cluster["fit"] = poly(ar_cluster["Year"], *param_1)
    
    #second fitting plot.
    ar_cluster.plot("Year", ["arable_k", "fit"])
    #setting title, labels.
    plt.title(f"{country} arable land %")
    plt.xlabel("Year")
    plt.ylabel("arable land %")
    #saving figure.
    plt.savefig("fit2.png", dpi = 300)

    
    
def forcast_data(fert_data_yw, ar_lnd_yw, country) :
    '''
    This function implements fitting algorithm to data and plots forcast graph.
    ----------
    fert_data_yw : pandas dataframe
        returns the pandas dataframe with specifed country.
    ar_lnd_yw : pandas dataframe
        returns the pandas dataframe with specifed country.
    country : string
        name of the country.

    Returns
    -------
    None.

    '''
    #saving the dataframe
    ar_cluster = pd.DataFrame()
    #selecting specific country.
    ar_cluster["fert_k"] = fert_data_yw[country]
    ar_cluster["arable_k"] = ar_lnd_yw[country]
    #dropping non zero values.
    ar_cluster = ar_cluster.dropna()
    #plotting the figure.
    plt.figure()
    #setting year as index.
    ar_cluster["Year"] = ar_cluster.index
    #parameter and covariance matrix. 
    param, covar = opt.curve_fit(poly, ar_cluster["Year"], ar_cluster["fert_k"])
    ar_cluster["fit"] = poly(ar_cluster["Year"], *param)
    #setting year range.
    year = np.arange(1995, 2025)
    #forcast function.
    forecast = poly(year, *param)
    sigma = err.error_prop(year, poly, param, covar)
    low = forecast - sigma
    up = forecast + sigma
    
    #plotting the figure.
    plt.figure()
    #forcast plot 1.
    plt.plot(ar_cluster["Year"], ar_cluster["fert_k"], label="Fertilizer consumption")
    plt.plot(year, forecast, label="forecast")
    #setting title, labels and legend.
    plt.title(f"{country} fertilizer consumption(kg per hectare of arable land)")
    plt.xlabel("Year")
    plt.ylabel("Fertilizer consumption(kg)")
    plt.legend()
    #plot uncertainty range.
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    #saving the figure.
    plt.savefig("forecast1.png", dpi = 300)

    #parameter and covariance matrix.
    param_1, covar_1 = opt.curve_fit(poly, ar_cluster["Year"], ar_cluster["arable_k"])
    ar_cluster["fit"] = poly(ar_cluster["Year"], *param_1)
    #setting year range.
    year = np.arange(1995, 2025)
    forecast = poly(year, *param_1)
    sigma = err.error_prop(year, poly, param_1, covar_1)
    low = forecast - sigma
    up = forecast + sigma

    #plotting the figure.
    plt.figure()
    plt.plot(ar_cluster["Year"], ar_cluster["arable_k"], label="arabel land %")
    plt.plot(year, forecast, label="forecast")
    #setting title, labels and legend.
    plt.title(f"{country} arable land %")
    plt.xlabel("Year")
    plt.ylabel("arable land %")
    plt.legend()

    #plot uncertainty range
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    #saving the figure.
    plt.savefig("forecast2.png", dpi = 300)

    # show all plots
    plt.show()
    

def poly(x, a, b, c, d):
    """ Calulates polynominal"""
    
    x = x - 1995
    f = a + b*x + c*x**2 + d*x**3
    
    return f


def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score

############## Main Function ################

#read csv files and get the dataframs.

fert_data_yw, fert_data_cw = \
    read_csv_file("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv")

ar_lnd_yw, ar_lnd_cw = read_csv_file("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_6302821.csv")
cr_lnd_yw, cr_lnd_cw = read_csv_file("API_AG.YLD.CREL.KG_DS2_en_csv_v2_6299928.csv")
#cluster plot.
clustering_data(fert_data_cw, cr_lnd_cw, ncluster= 2, year_1= 1995, year_2= 2020)
#fitting plot.
fit_data(fert_data_yw, ar_lnd_yw, country = "China") 
#forcast plot.
forcast_data(fert_data_yw, ar_lnd_yw, country = "China")



