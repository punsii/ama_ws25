{ lib, ... }:
{
  imports = [
    ./ama.nix
  ];

  ama.enable = lib.mkDefault true;
}
