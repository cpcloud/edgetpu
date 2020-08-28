self: super:
{
  opencv4 = (
    super.opencv4.override {
      enableContrib = false;
      enableCuda = false;
      enableEigen = false;
      enableEXR = false;
      enableWebP = false;
      enableOpenblas = false;
      enableIpp = false;
      enableFfmpeg = false; # camd_mock needs this for videoio; TODO: remove when sensord_mock exists
      enableGStreamer = false;
      enableGtk2 = false;
      enableGtk3 = true;
      enableTesseract = false;
      enableTbb = false;
      enableVtk = false;
      enableOvis = false;
      enableGPhoto2 = false;
      enableDC1394 = false;
      enableJPEG2K = false;
      enableJPEG = false; # needed for the pak file writer
      enablePNG = false; # sensord_snap uses imwrite to produce PNGs; TODO: look into the image crate to replace this
      enablePython = false;
      enableTIFF = false;
      enableDocs = false;
    }
  ).overrideAttrs (
    attrs:
    let
      lib = super.lib;
    in
    {
      buildInputs = lib.remove self.protobuf attrs.buildInputs;
      cmakeFlags = [
        "-DWITH_PTHREADS_PF=OFF"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_PERF_TESTS=OFF"
        "-DBUILD_TESTS=OFF"
        "-DBUILD_EXAMPLES=OFF"
        "-DBUILD_opencv_apps=OFF"
        "-DWITH_OPENCL=OFF"
        "-DWITH_LAPACK=OFF"
        "-DWITH_ITT=OFF"
        "-DWITH_PROTOBUF=OFF"
        "-DWITH_IMGCODEC_HDR=OFF"
        "-DWITH_IMGCODEC_SUNRASTER=OFF"
        "-DWITH_IMGCODEC_PXM=OFF"
        "-DWITH_IMGCODEC_PFM=OFF"
        "-DWITH_QUIRC=OFF"
        "-DWITH_WEBP=OFF"
        "-DBUILD_LIST=videoio,highgui"
      ] ++ lib.filter (f: !lib.strings.hasInfix "WITH_OPENMP" f) (lib.flatten attrs.cmakeFlags);
    }
  );
}
