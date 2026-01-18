{ python, ... }:
python.pkgs.buildPythonPackage {
  pname = "ama_tlbx";
  version = "0.1.0";
  pyproject = true;
  src = ../../ama_tlbx;
  # depenencies = [];
  propagatedBuildInputs = with python.pkgs; [
    devtools
    hatchling
    matplotlib
    numpy
    pandas
    patsy
    plotly
    pycountry
    pytest
    scikit-learn
    scipy
    seaborn
    statsmodels
    streamlit
    sympy
  ];
}
