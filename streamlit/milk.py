from pathlib import Path

import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

import streamlit as st

DATA_DIR = Path("_data")
assert DATA_DIR.exists(), DATA_DIR

st.set_page_config(
    page_title="Angewandte Multivariate Analysemethoden",
    page_icon="",
    layout="wide",
    # initial_sidebar_state="expanded",
)


st.markdown(
    "[![GitHub](https://img.shields.io/badge/github-%2523121011.svg?style=for-the-badge&logo=github&color=AB00AB)](https://github.com/punsii/ama_ws25)"
)

df = pd.read_csv(DATA_DIR / "milk.csv", sep=";", decimal=",")
# float_columns = ["water", "protein", "fat", "lactose", "ash"]
# for column in float_columns:
#     df[column] = df[column].astype(float)

n = 1000
m = KMeans(5)
m.fit(df.drop(columns=["name"]))

df["cl"] = m.labels_
st.write(df)
st.write(df.plot.scatter("water", "protein", c="cl", colormap="gist_rainbow").figure)

# st.write(
#     sns.pairplot(
#         df.drop(columns=["name"]),
#         height=3,
#     ).figure
# )
# st.dataframe(
#     df.drop(columns=["name"])
#     .corr()
#     .style.background_gradient(vmin=-1, vmax=1, cmap="RdYlGn")
# )


# fig = plt.figure(1, figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# pca = PCA(n_components=2)
# pca = pca.fit(df.drop(columns=["name"]))
# st.write(pca.components_)
# scatter = ax.scatter(
#     pca[:, 0],
#     pca[:, 1],
#     pca[:, 2],
#     s=40,
# )
#
# ax.set(
#     title="First three PCA dimensions",
#     xlabel="1st Eigenvector",
#     ylabel="2nd Eigenvector",
#     zlabel="3rd Eigenvector",
# )
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# st.write(fig)

# # Add a legend
# legend1 = ax.legend(
#     scatter.legend_elements()[0],
#     iris.target_names.tolist(),
#     loc="upper right",
#     title="Classes",
# )
# ax.add_artist(legend1)

# plt.show()


df = df.drop(columns=["name"])

fig = plt.figure(figsize=(5, 5))
Z = linkage(df, "ward")
dn = dendrogram(Z)
st.write(fig)

fig = plt.figure(figsize=(5, 5))
Z = linkage(df, "single")
dn = dendrogram(Z)
st.write(fig)

fig = plt.figure(figsize=(5, 5))
Z = linkage(df, "complete")
dn = dendrogram(Z)
st.write(fig)
