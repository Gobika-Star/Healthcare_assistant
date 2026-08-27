{ pkgs }: {
  channel = "stable-23.11";
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.tesseract            # OCR engine support
    pkgs.ffmpeg               # Audio processing for voice features
    pkgs.libglvnd             # Required for OpenCV execution
    pkgs.glib
  ];
  idx = {
    extensions = [
      "ms-python.python"
      "ms-azuretools.vscode-docker"
    ];
    workspace = {
      onCreate = {
        install-deps = "cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt";
      };
    };
  };
}