import pandas as pd
import statistics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # To visualize
import numpy as np
from sklearn.metrics import r2_score
import scipy

df = pd.read_csv('C:\\Users\\Laptop User 8\\Desktop\\all-ages.csv')
df = df.astype(dtype={'Major_category': 'category'})
df = df.astype(dtype={'Major': 'category'})
# df.info()
# df.head()
# df.describe()
# print(df.shape)

# Questions: *Does major category impact level of employment?(Average,Min/Max)*   *Which major has highest employment?(Relative & Absolute)*
# *Major/category with highest/minimum employment/unemployment?*    *Which majors have high/low salaries based on median, 25th, 75percentile*
# *The biggest disparity of the empl/unemployed for category(find stability?)*    *Which are the most/least popular major & category (students enrolled)*

# **What are the most/least popular major & category (students enrolled)?**
print(df["Total"].max())
print(df["Total"].min())
major_totals = df.groupby("Major")["Total"].sum().sort_values()
print(major_totals.head())
print(major_totals.tail())
major_totals.head(20).plot(kind="barh", fontsize=6, title="20 Least Popular Majors Based On Student Enrollment")
plt.show()
major_totals.tail(20).plot(kind="barh", fontsize=6, title="Top 20 Most Popular Majors Based On Student Enrollment")
plt.show()
cats_totals = df.groupby("Major_category")["Total"].sum().sort_values()
cats_totals.plot(kind="barh", fontsize=6, title="Major Categories Student Enrollment")
plt.show()
# print("***************************************************************************************************")


# *Does Major Category affect Employability?*
cats_emp = df.groupby("Major_category")["Employed"].mean().sort_values()
cats_emp.plot(kind="barh", fontsize=6, title="Average Employed Graduates From Each Major Category Compared")
cats_ue = df.groupby("Major_category")["Unemployed"].mean().sort_values()
cats_ue.plot(kind="barh", fontsize=6, title="Average Unemployed Graduates From Major Categories Compared")
plt.show()
# how do I plot this?*  -- make 4 different graphs  --------------------- is this correct? why was it major/major cat?
cats_emp_min = df.groupby(["Major_category", "Major"],as_index=False)["Employed"].min()
print(cats_emp_min.dropna())
for row_label, row in cats_emp_min.dropna().iterrows():
        print(row_label, row, sep='\n', end='\n\n')
print(cats_emp_min.head)
cats_emp_min.plot(kind="bar", fontsize=5)
plt.show()
*** or is this correct??
cats_emp_max = df.groupby("Major_category")["Employed"].max()
cats_emp_max.dropna()
cats_emp_max.dropna().plot(kind="barh", fontsize=5)
plt.show()
print(cats_emp_max)
cats_ue_min = df.groupby("Major_category")["Unemployed"].min()
cats_ue_min.dropna()
cats_ue_min.dropna().plot(kind="barh", fontsize=5)
plt.show()
cats_ue_max = df.groupby("Major_category")["Unemployed"].max()
cats_ue_max.dropna()
cats_ue_max.dropna().plot(kind="barh", fontsize=5)
plt.show()
print(cats_emp_min.head())
print(cats_emp_max)
print(cats_ue_min)
print(cats_ue_max)
plt.show()



# *Which majors & categories have high/low salaries based on median, 25th, 75percentile salaries?*
cats_25 = df.groupby("Major_category")["P25th"].mean()
cats_med = df.groupby("Major_category")["Median"].mean()
cats_75 = df.groupby("Major_category")["P75th"].mean()
cats_salary = pd.concat([cats_25,cats_med,cats_75], axis= 1)
cats_salary["Major_category"] = cats_salary.index
cats_salary.index = range(0,len(cats_salary))
print(cats_salary.head())
cats_salary.plot(x="Major_category",y=["P25th","Median","P75th"],kind="barh",title="Major Category Salaries based on Median, 25th & 75th percentiles",fontsize=6)
print(list(cats_25.index))
plt.show()
df.hist(column="Median")
^*median salary peaks around 45-50k also shows that there are majors who earn more
top_10 = df.sort_values(by="Median", ascending=False).head(10)
top_10.plot(x="Major", y="Median", kind="bar", rot=11, fontsize=5, title="Top 10 Median Salaries by Major")
top_medians = df[df["Median"] > 80000].sort_values("Median")
top_medians.plot(x="Major", y=["P25th", "Median", "P75th"], kind="bar", rot=7, fontsize=6, title= "Salaries for Majors earning over 80k")
# ^investigating outliers
plt.show()
# print("***************************************************************************************************")


# *does this show/prove anything?*
cats_emp= df.groupby("Major_category", as_index=False)["Employed"].mean()
print(cats_emp.head())
cats_ue = df.groupby("Major_category", as_index=False)["Unemployed"].mean()
med_emp = df.groupby("Major_category", as_index=False)["Median"].mean()
print(med_emp)
features = np.dstack((cats_emp["Employed"],med_emp["Median"]))
print(features[0])
plt.scatter(x=cats_emp["Employed"],y=med_emp["Median"])
plt.show()


plt.scatter(x=df["Employed"]/df["Total"],y=df["Median"])
features = np.dstack((df["Employed"]/df["Total"], df["Median"]))
print(features[0])
plt.show()
kmeans = KMeans(
    init="random",
    n_clusters=2,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(features[0])
print("kmeans.inertia_ \n",kmeans.inertia_)
print("kmeans.cluster_centers_ \n",kmeans.cluster_centers_)
print("kmeans.n_iter_ \n",kmeans.n_iter_)
kmeans.labels_[:]
klabels = kmeans.labels_
print(klabels)
center = np.array(["yellow","blue"])
print(center)
plt.scatter(df["Employed"]/df["Total"],df["Median"],c=klabels)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c=center,marker="*",s=300)
plt.show()
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(features[0])
    sse.append(kmeans.inertia_)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(features[0])
    score = silhouette_score(features[0], kmeans.labels_)
    silhouette_coefficients.append(score)
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
# print("***************************************************************************************************")

perc_emp = df["Employed"]/df["Total"]
plt.hist(df["Employed"]/df["Total"], bins=15)
plt.show()

mean_rand_tot = []
for i in range(10000):
    rand_tot = perc_emp.sample(n=40)
    mean_rand_tot.append(rand_tot.mean())

plt.hist(mean_rand_tot,bins=13*3)
plt.title("PDF of Percentage Employed in each Major")


mean_rand_tot = np.array(mean_rand_tot)
counts, bins = np.histogram(mean_rand_tot)
mids = 0.5*(bins[1:] + bins[:-1])
probs = counts / np.sum(counts)

mean = np.sum(probs * mids)
sd = np.sqrt(np.sum(probs * (mids - mean)**2))

plt.text(23, 45, f'$\mu={mean}, sd={sd}$')
plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
plt.axvline(mean-sd, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mean+sd, color='r', linestyle='dashed', linewidth=2)

plt.figure(figsize=(20,20),facecolor='white',edgecolor='blue')

plt.show()

x = np.sort(mean_rand_tot)
y = np.arange(len(mean_rand_tot)) / float(len(mean_rand_tot))
y_pdf = counts / sum(counts)
plt.xlabel('% Employed')
plt.ylabel('Probability of being employed <= x')

plt.title('CDF using sorting the data')

plt.plot(x, y, marker='')
plt.plot(bins[1:], y_pdf)
plt.show()


x = df["Unemployed"]/df["Total"]
y = df["Median"]
x = df["Employed"]/df["Total"]
y = df["Median"]
plt.scatter(x,y)
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
model = LinearRegression()  # create object for the class
model.fit(x, y)  # perform linear regression
Y_pred = model.predict(x)  # make predictions
plt.plot(x, Y_pred, color='red')
plt.show()
r_sq = model.score(x, y)
r2 = r2_score(y, model.predict(x.reshape(-1,1)))
chisq = scipy.stats.chisquare(y, Y_pred)

print('coefficient of determination:', r_sq)
print(f'r2_score = {r2}')
print(f'chisq = {chisq}')

# Pandas Correlation
x = df["Unemployed"]/df["Total"]
y = df["Median"]
x = df["Employed"]/df["Total"]
y = df["Median"]
corr = x.corr(y)
print(f'Correlation = {corr}')
