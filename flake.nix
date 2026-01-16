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
        overlays = [
          (final: prev: {
            quarto = prev.quarto.override {
              extraPythonPackages = pyPkgs;
            };
          })
        ];
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

      ama_tlbx = pkgs.callPackage ./nix/pkgs/ama_tlbx.nix {
        inherit python;
      };
      python = pkgs.python3 // {
        pkgs = pkgs.python3.pkgs.overrideScope (
          self: super: {
            inherit ama_tlbx;
          }
        );
      };
      pyPkgs =
        pythonPackages: with pythonPackages; [
          ama_tlbx
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
        ];
      pythonEnv = python.withPackages pyPkgs;

      src = pkgs.nix-gitignore.gitignoreSource [ ] ./.;
      runDev = pkgs.writeShellApplication {
        name = "runDev";
        text = "${pythonEnv}/bin/python3 -m streamlit run ./src/app.py";
      };
      runProd = pkgs.writeShellApplication {
        name = "runProd";
        text = "${pythonEnv}/bin/python3 -m streamlit run ${src}/src/app.py";
      };
      QUARTO_PYTHON = "${pythonEnv}/bin/python";
      DATA_DIR = ./_data;

      submission = pkgs.stdenv.mkDerivation {
        name = "submission";
        src = ./.;
        inherit QUARTO_PYTHON DATA_DIR;
        buildPhase = ''
          cd submission
          export HOME=$(mktemp -d)
          # export QUARTO_PYTHON="${pythonEnv}/bin/python"

          mkdir -p $out/var/www/ama/
          ${pkgs.quarto}/bin/quarto render .
          mv _site $out/var/www/ama/
          find $out/var/www/ama/
        '';
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
        inherit pythonEnv submission;
        default = submission;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          treefmtEval.config.build.wrapper
          pythonEnv
          quarto
        ];
        inherit QUARTO_PYTHON DATA_DIR;
      };

      formatter.${system} = treefmtEval.config.build.wrapper;
    };
}
