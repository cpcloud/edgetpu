self: super: {
  python-flatbuffers = self.python3Packages.buildPythonPackage {
    pname = "python-flatbuffers";
    inherit (self.flatbuffers) version meta;
    src = "${self.flatbuffers.src}/python";
    propagatedNativeBuildInputs = [ self.flatbuffers ];
  };
}
