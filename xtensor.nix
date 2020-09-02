self: super: rec {
  xtensor = self.stdenv.mkDerivation rec {
    pname = "xtensor";
    version = "0.21.5";

    src = self.fetchFromGitHub {
      owner = "xtensor-stack";
      repo = "xtensor";
      rev = version;
      sha256 = "0nqhma6ps2pypchlmcd99rwq9r2qlj3v3rwyxvdb8iawgkvc0905";
    };

    nativeBuildInputs = [ self.cmake ];
    propagatedBuildInputs = [ xtl ];
  };

  xtl = self.stdenv.mkDerivation rec {
    pname = "xtl";
    version = "0.6.16";

    src = self.fetchFromGitHub {
      owner = "xtensor-stack";
      repo = "xtl";
      rev = version;
      sha256 = "0hkz01l7fc1m79s02hz86cl9nb4rwdvg255r6aj82gnsx5qvxy2l";
    };

    nativeBuildInputs = [ self.cmake ];
  };

  xsimd = self.stdenv.mkDerivation rec {
    pname = "xsimd";
    version = "7.4.8";

    src = self.fetchFromGitHub {
      owner = "xtensor-stack";
      repo = "xsimd";
      rev = version;
      sha256 = "1bk9cn7sd3zipfq9jg1bmnanxii6f046p2x8y1pww3qxps03k9fa";
    };

    nativeBuildInputs = [ self.cmake ];
    propagatedBuildInputs = [ xtl ];
  };

  xtensor-io = self.stdenv.mkDerivation rec {
    pname = "xtensor-io";
    version = "0.9.0";

    src = self.fetchFromGitHub {
      owner = "xtensor-stack";
      repo = "xtensor-io";
      rev = version;
      sha256 = "0nsspmxq79191f8dzsl5f8lv0z759dv1a7794mdx2ki4nj6z32yx";
    };

    nativeBuildInputs = [ self.cmake ];
    propagatedBuildInputs = [ xtl xtensor self.openimageio2 self.openexr ];
  };

  tinyformat = self.stdenv.mkDerivation rec {
    pname = "tinyformat";
    version = "v2.3.0";

    src = self.fetchFromGitHub {
      owner = "c42f";
      repo = "tinyformat";
      rev = version;
      sha256 = "0j8bxrg7i4rn98iqpfh03krshgcbwh8hq8nm0p3sfdqpira3vi3y";
    };

    dontBuild = true;

    installPhase = ''
      mkdir -p $out/include
      cp $src/tinyformat.h $out/include
    '';

    nativeBuildInputs = [ self.cmake ];
  };
}
