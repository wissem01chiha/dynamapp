function rmbd_make(target, varargin)
% Build tool for mypackage
%
% rmbd_make <target> [...options...]
%
% This is the main build tool for doing all the build and packaging operations
% for mypackage. It's intended to be called as a command. This is what you will
% use to build & package the distribution files for a release of the package.
%
% Targets:
%
%   test      - run the tests
%
%   dist      - build the full dist files (both zips and toolbox)
%   zips      - build the dist archive files (zips)
%
%   toolbox   - build the Matlab Toolbox .mltbx installer file
%   clean     - delete all the derived artifacts
%
%   docs      - build docs/ etc. (the GH Pages stuff) from docs-src and examples (merge)
%   doc       - build doc/, the final static (local) doco files (from docs/, replace)
%   m-doc     - build build/M-doc/ MLTBX format docs (from doc/)
%   docview   - live-preview the project doco (from docs/) (requires Jekyll)
%
%   build     - "build" (transform and pcode) the source code
%   buildmex  - build all the MEX files in the source tree
%
%   simplify  - remove some optional MatlabProjectTemplate features from this repo
%   util-shim <package> - "shim" utility functions into a package

%#ok<*STRNU>

arguments
  target (1,1) string
end
arguments (Repeating)
  varargin
end


if target == "test"
  mypackage_launchtests
elseif target == "docs"
  build_docs
elseif target == "doc"
  build_doc
elseif target == "doc-preview"
  preview_docs
elseif target == "m-doc"
  make_mdoc
elseif target == "toolbox"
  mypackage_make m-doc
  mypackage_package_toolbox
elseif target == "zips"
  mypackage_make build
  mypackage_make m-doc
  make_zips
elseif target == "dist"
  mypackage_make zips
  mypackage_make toolbox
  fprintf('Made dist.\n')
elseif target == "clean"
  make_clean
elseif target == "build"
  mypackage_build
elseif target == "buildmex"
  mypackage_build_all_mex
elseif target == "simplify"
  make_simplify
elseif target == "util-shim"
  pkg = varargin{1};
  make_util_shim(pkg);
else
  error("Unknown target: %s", target);
end

end

function make_mdoc
rmrf('build/M-doc')
mkdir2('build/M-doc')
copyfile2('doc/*', 'build/M-doc')
if isfile('build/M-doc/feed.xml')
  delete('build/M-doc/feed.xml')
end
end

function preview_docs
import mypackage.internal.util.*;
RAII.cd = withcd('docs');
make_doc --preview
end

function make_zips
program = "myproject";
distName = program + "-" + mypackage.globals.version;
verDistDir = fullfile("dist", distName);
distfiles = ["build/Mcode" "doc" "lib" "examples" "README.md" "LICENSE" ...
  "CHANGES.md" "VERSION"];
rmrf([verDistDir, distName+".tar.gz", distName+".zip"])
if ~isfolder('dist')
  mkdir2('dist')
end
mkdir2(verDistDir)
for distFile = distfiles
  if isfile(distFile) || isfolder(distFile)
    system2(sprintf("cp -R '%s' '%s'", distFile, verDistDir));
  end
end
RAII.cd = withcd('dist');
tar(distName+".tar.gz", distName)
zip(distName+".zip", distName)
end

function make_clean
rmrf(strsplit("dist/* build docs/site docs/_site test-output", " "));
end

function make_simplify
rmrf(strsplit(".circleci .travis.yml azure-pipelines.yml src lib/java/MyCoolProject-java", " "));
end

function make_package_docs(varargin)
doOnlySrc = ismember('--src', varargin);
build_docs;
if ~doOnlySrc
  build_doc;
end
end

function build_docs
% Build the generated (Markdown) parts of the doc sources in docs/
RAII.cd = withcd(reporoot);
docsDir = fullfile(reporoot, 'docs');
% Copy over examples
docsExsDir = fullfile(docsDir, 'examples');
if isfolder(docsExsDir)
  rmdir2(docsExsDir, 's');
end
copyfile('examples', fullfile('docs', 'examples'));
% TODO: Generate API Reference
end

function build_doc
% Build the final static local doc files in doc/
RAII.cd = withcd(fullfile(reporoot, 'docs'));
make_doc;
delete('../doc/make_doc*');
end

function make_util_shim(pkg)
shimsDir = fullfile(reporoot, 'dev-kit', 'util-shims');
relpkgpath = strjoin(strcat("+", strsplit(pkg, ".")), "/");
pkgdir = fullfile(fullfile(reporoot, 'Mcode'), relpkgpath);
if ~isfolder(pkgdir)
  error('Package folder does not exist: %s', pkgdir);
end
privateDir = fullfile(pkgdir, 'private');
if ~isfolder(privateDir)
  mkdir(privateDir);
end
copyfile2(fullfile(shimsDir, '*.m'), privateDir);
fprintf('Util-shimmed package: %s', pkg);
end
