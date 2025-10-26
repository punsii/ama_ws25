{
  description = "Angewandte Multivariate Analysemethoden";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };
  outputs =
    {
      nixpkgs,
      treefmt-nix,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      treefmtEval = treefmt-nix.lib.evalModule pkgs {
        # Used to find the project root
        projectRootFile = "flake.nix";

        programs = {
          ruff-check.enable = true;
          ruff-format.enable = true;
          black.enable = true;
          isort.enable = true;
          prettier.enable = true;
          nixfmt.enable = true;
        };
      };
      pythonEnv = pkgs.python3.withPackages (
        ps: with ps; [
          ipykernel
          ipympl
          ipython
          jupyter
          jupyterlab
          jupyterlab-widgets
          matplotlib
          mypy
          nbformat
          notebook
          numpy
          pandas
          pandas-stubs
          pdoc
          pytest
          pytest-cov
          ruff
          scikit-learn
          scipy
          seaborn
          statsmodels
          streamlit
          sympy
        ]
      );

      src = pkgs.nix-gitignore.gitignoreSource [ ] ./.;
      runDev = pkgs.writeShellApplication {
        name = "runDev";
        text = ''${pythonEnv}/bin/python3 -m streamlit run ./src/app.py'';
      };
      runProd = pkgs.writeShellApplication {
        name = "runProd";
        text = ''${pythonEnv}/bin/python3 -m streamlit run ${src}/src/app.py'';
      };
    in
    {
      apps.${system} = rec {
        dev = {
          type = "app";
          program = "${runDev}/bin/runDev";
        };
        prod = {
          type = "app";
          program = "${runProd}/bin/runProd";
        };
        default = dev;
      };

      packages.${system} = rec {
        inherit pythonEnv;
        default = pythonEnv;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          treefmtEval.config.build.wrapper
          pythonEnv
          quarto
        ];
      };

      formatter.${system} = treefmtEval.config.build.wrapper;
    };
}
