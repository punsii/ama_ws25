{ python, ... }:
python.pkgs.buildPythonPackage {
  pname = "ama_tlbx";
  version = "0.1.0";
  pyproject = true;
  src = ../../ama_tlbx;
  # depenencies = [];
  propagatedBuildInputs = with python.pkgs; [
    hatchling
    matplotlib
    numpy
    pandas
    pytest
    scikit-learn
    scipy
    seaborn
    sympy
  ];
}
