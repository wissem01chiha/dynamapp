function load_mypackage

repoDir = fileparts(fileparts(mfilename('fullpath')));
toplevelDir = [repoDir '/Mcode'];
addpath(toplevelDir);
