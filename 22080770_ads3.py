# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 01:48:31 2024

@author: OMKAR
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:00:19 2023

@author: OMKAR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cluster_tools as clt
import sklearn.cluster as sk

def file_name(file):
    '''
    This function reads the csv files and also filters the dataframe \
    and returns two dataframe one with transposed data and one with
    non transposed data.

    Parameters
    ----------
    file : TYPE
        This is pandas dataframe.

    Returns
    -------
    yr_wdf : TYPE
        Returns the transposed dataframe.
    cr_wdf : TYPE
        Returns non transposed dataframe.

    '''
    # reading csv files and selecting specific years.
    wdf = agriculture_land = pd.read_csv(file, skiprows=3,
                                         usecols=["Country Name", "1990", "1991",
                                                  "1992", "1993", "1994", "1995",
                                                  "1996", "1997", "1998", "1999",
                                                  "2000", "2001", "2002", "2003",
                                                  "2004", "2005", "2006", "2007",
                                                  "2008", "2009", "2010", "2011",
                                                  "2012", "2013", "2014", "2015",
                                                  "2016", "2017", "2018", "2019",
                                                  "2020"])
    # setting country name as index.
    wdf.index = wdf["Country Name"]
    wdf = agriculture_land.iloc[:, 1:]

    # filtering data.
    countires = ["France", "India", "Russian Federation", "Netherlands",
                 "Hungary", "Germany", "Australia"]
    cleaned_agriculture = wdf.loc[countires, :]

    # setting years as intengers.
    cleaned_agriculture.columns = cleaned_agriculture.columns.astype(int)

    ''' Separate the data into two dataframes: one with years as columns
    and one with countries as columns  '''
    yr_wdf = cleaned_agriculture.T
    cr_wdf = cleaned_agriculture

    return yr_wdf, cr_wdf

df_fert = pd.read_csv("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv", skiprows=(1,1))
df_agri = pd.read_csv("API_SL.AGR.EMPL.FE.ZS_DS2_en_csv_v2_6302174.csv", skiprows=(1,1))
print(df_fert)
print(df_agri)

clt.map_corr(df_fert)
def heat_map(countires, df_fert, df_agri, cmap):
  
    # Corelation heatmap indicators.
    cor = pd.DataFrame()
    cor["Agriculture"] = cr_agriculture.loc[countires, :].values
    cor["forest"] = cr_forest.loc[countires, :].values
    cor["co2"] = cr_co2.loc[countires, :].values
    cor["urban population"] = cr_pop.loc[countires, :].values
    cor["Renewable Energy"] = cr_energy.loc[countires, :].values
    cor["Mortality rate"] = yr_mortality.loc[countires, :].values

    # correlation calculation.
    cor = cor.corr().round(3)
    # plotting the figure.
    plt.figure()
    # colour map.
    plt.imshow(cor, cmap=cmap)
    # adding colour bar.
    plt.colorbar()
    # adding x-ticks and y-ticks.
    plt.xticks(np.arange(len(cor.columns)), labels=cor.columns,
               rotation=90)
    plt.yticks(np.arange(len(cor.columns)), labels=cor.columns)

    plt.title(countires)
    for (i, j), ra_r in np.ndenumerate(cor):
        plt.text(i, j, ra_r, ha="center", va="center")
    # Saving the figure.
    plt.savefig(countires+".png", dpi=300, bbox_inches="tight")
    plt.show()


df_fish_new = df_fish[["fork length", "height"]].copy()
print(df_fish_new)

df_norm = clt.scaler(df_fish_new)
df_normal = df_norm[0]

df_min = df_norm[1]
df_max = df_norm[2]

print(df_min)
print(df_max)

#centres = [[-1., 0.], [1., -0.5], [0., 1.]]



plt.figure()
cm = plt.colormaps["Paired"]
plt.scatter(df_normal["fork length"], df_normal["height"], marker="o", cmap=cm)
plt.show()
nclust = 4
kmeans = sk.KMeans(n_clusters = nclust, n_init=20)
kmeans.fit(df_normal)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
print(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure()
plt.scatter(df_normal["fork length"], df_normal["height"], 10, labels, marker="o", cmap=cm)
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label="kmeans centres")
plt.scatter(xkmeans, ykmeans, 45, "y", marker="+", label="real centres")
plt.show()

#main
yr_fert,cr_fert = file_name("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv")
yf_agri,cr_agri = file_name("API_SL.AGR.EMPL.FE.ZS_DS2_en_csv_v2_6302174.csv")
print(df_fert)
print(df_agri)

