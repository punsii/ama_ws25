{
  pkgs,
  lib,
  config,
  ...
}:
{

  options = {
    ama.enable = lib.mkEnableOption "enables applied multivariate analysis streamlit service";
  };

  config = lib.mkIf config.ama.enable {
    systemd.timers."ama" = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "*-*-* 03:30:00";
        RandomizedDelaySec = "1800";
        Persistent = "true";
        Unit = "ama-restart";
      };
    };
    systemd.services = {
      "ama-restart" = {
        description = "Service for restarting the applied multivariate analysis streamlit app";
        script = ''
          ${pkgs.systemd}/bin/systemctl restart ama.service
        '';
        wantedBy = [ "multi-user.target" ];
        after = [ "network-online.target" ];
        serviceConfig = {
          Type = "oneshot";
        };
      };
      "ama" =
        let
          WorkingDirectory = "/root/uni_services/ama/";
        in
        {
          description = "Service for hosting the applied multivariate analysis streamlit app";
          script = ''
            mkdir -p ${WorkingDirectory};
            ${pkgs.nix}/bin/nix run "git+ssh://gitlab.lrz.de/XXXXXXXXXXXXXXXXXXXXXXX"
          '';
          wantedBy = [ "multi-user.target" ];
          after = [ "network-online.target" ];
          serviceConfig = {
            inherit WorkingDirectory;
          };
        };
    };

    services.caddy = {
      enable = true;
      globalConfig = ''
        email paul@menhart.net
      '';

      virtualHosts."menhart-testing.duckdns.org".extraConfig = ''
        encode gzip
        reverse_proxy 127.0.0.1:13338
      '';
    };

    networking.firewall.allowedTCPPorts = [
      80
      443
    ];
  };
}
