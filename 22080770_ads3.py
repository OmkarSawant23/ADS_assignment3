# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 04:13:23 2024

@author: OMKAR
"""



import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors as err

def read_world_bank_csv(filename):

    # set year range and country list to filter the dataset
    start_from_yeart = 1995
    end_to_year = 2020
    #countrie_list = ["China","Russian Federation","India"]

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
    df_country_index = wb_df.loc[
      #  wb_df["Country Name"].isin(countrie_list),
        :,all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index

def poly(x, a, b, c, d):
    """ Calulates polynominal"""
    
    x = x - 1990
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

###### Main Function ################

# read csv files and get the dataframs

fert_data_yw, fert_data_cw = \
    read_world_bank_csv("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv")

ar_lnd_yw, ar_lnd_cw = read_world_bank_csv("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_6302821.csv")
cr_lnd_yw, cr_lnd_cw = read_world_bank_csv("API_AG.YLD.CREL.KG_DS2_en_csv_v2_6299928.csv")

#print(ar_lnd_cw[1995])
###############Clustering   ######################################

ar_cluster = pd.DataFrame()
ar_cluster["fert"] = fert_data_cw.loc[:,1995]
ar_cluster["arable"] = cr_lnd_cw.loc[:,1995]
ar_cluster["fert_2000"] = fert_data_cw.loc[:,2020]
ar_cluster["arable_2000"] = cr_lnd_cw.loc[:,2020]
ar_cluster = ar_cluster.dropna()

print(ar_cluster)
df_norm, df_min, df_max = ct.scaler(ar_cluster)
ncluster = 3

for ic in range(2, 11):
    score = one_silhoutte(ar_cluster, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs


labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = ct.backscale(cen, df_min, df_max)

xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# extract x and y values of data points
x1 = ar_cluster["fert"]
y1 = ar_cluster["arable"]
x2 = ar_cluster["fert_2000"]
y2 = ar_cluster["arable_2000"]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
cm = plt.colormaps["Paired"]
fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
scatter1 = axs[0].scatter(x1, y1, 10, labels, marker="o", cmap=cm)
axs[0].scatter(xkmeans, ykmeans, 45, "k", marker="d")
axs[0].scatter(xkmeans, ykmeans, 45, "k", marker="d")
axs[0].set_xlabel("Fertilizer(kg)")
axs[0].set_ylabel("Cereal")
axs[0].set_title("1995")
axs[0].set_ylim(0, 30000)
axs[0].grid(True)
scatter2 = axs[1].scatter(x2, y2, s=10, c=labels, marker="o", cmap=cm)
axs[1].scatter(xkmeans, ykmeans, 45, "k", marker="d")
axs[1].scatter(xkmeans, ykmeans, 45, "k", marker="d")
axs[1].set_xlabel("Fertilizer(kg)")
axs[1].set_ylabel("Cereal")
axs[1].set_title("2022")
axs[1].set_ylim(0, 30000)
axs[1].grid(True)
# show cluster centres
#plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
#plt.scatter(xkmeans, ykmeans, 45, "y", marker="+")
#plt.title("China Arable land % vs fertilizer consumption(kg per hectare of arable land)")
#plt.xlabel("Fertilizer consumption(kg)")
#plt.ylabel("arable land %")
#####################fitting############################


plt.figure()
ar_cluster["Year"] = ar_cluster.index


param, covar = opt.curve_fit(poly, ar_cluster["Year"], ar_cluster["fert"])
ar_cluster["fit"] = poly(ar_cluster["Year"], *param)

ar_cluster.plot("Year", ["fert", "fit"])


############Forcast###############
year = np.arange(1995, 2025)
forecast = poly(year, *param)
sigma = err.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma

ar_cluster["fit"] = poly(ar_cluster["Year"], *param)

plt.figure()
plt.plot(ar_cluster["Year"], ar_cluster["fert"], label="Fertilizer consumption")
plt.plot(year, forecast, label="forecast")
plt.title("China fertilizer consumption(kg per hectare of arable land)")
plt.xlabel("Year")
plt.ylabel("Fertilizer consumption(kg)")

# plot uncertainty range
plt.fill_between(year, low, up, color="yellow", alpha=0.7)


# show all plots
plt.show()