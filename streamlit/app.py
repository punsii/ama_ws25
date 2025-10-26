from pathlib import Path

import pandas as pd
import seaborn as sns

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

df = (
    pd.read_csv(DATA_DIR / "life_expectancy_data.csv")
    .pipe(
        lambda d: d.set_axis(
            d.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("__+", "_", regex=True),
            axis=1,
        )
    )
    .assign(
        year=lambda d: pd.to_datetime(d.year.astype(str), format="%Y", errors="coerce"),
    )
    .pipe(
        lambda d: d.assign(
            **{
                col: pd.to_numeric(d[col], errors="coerce")
                for col in d.columns
                if col not in {"country", "status", "year"}
            }
        )
    )
)


st.write(df.dtypes)
st.write(df)

df_numeric = df.drop(columns=["country", "status", "year", "life_expectancy"])
st.dataframe(
    df_numeric.corr().style.background_gradient(vmin=-1, vmax=1, cmap="RdYlGn")
)
st.write(df_numeric.corrwith(df["life_expectancy"]))

# st.write("General PCA:")
# pca = PCA(n_components=2)
# pca.fit(df_numeric)
# st.write(pca.components_)
# st.write(
#     pd.DataFrame(pca.get_covariance()).style.background_gradient(
#         vmin=-1, vmax=1, cmap="RdYlGn"
#     )
# )
#
# st.write("thinnes PCA:")
# pca2 = PCA(n_components=2)
# pca2.fit(df_numeric.drop(columns=["thinness_1-19_years", "thinness_5-9_years"]))
# st.write(pca2.components_)
# st.write(
#     pd.DataFrame(pca2.get_covariance()).style.background_gradient(
#         vmin=-1, vmax=1, cmap="RdYlGn"
#     )
# )
st.write(
    sns.pairplot(
        df.drop(
            columns=[
                "country",
                "year",
                "life_expectancy",
                "infant_deaths",
                "percentage_expenditure",
                "thinness_5-9_years",
            ]
        ),
        hue="status",
        height=10,
    ).figure
)
