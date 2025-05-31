function initializePackage
% Basic package initialization
%
% This should *only* do basic library initialization involving paths and dependency
% loading and the like. It should *not* discover initial values for library settings;
% that's done in mypackage.Settings.discover. It has to be that way so the package
% settings state can handle a `clear classes` gracefully.

% Do not re-initialize if already initialized
if mypackage.internal.misc.getpackageappdata('initialized')
  return
end

% Don't depend on globals, to avoid circular dependency
% distroot = mypackage.globals.distroot;
distroot = string(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));

% Java dependencies
libJava = fullfile(distroot, 'lib', 'java');
if isfolder(libJava)
  javaLibs = readdir(libJava);
  for jlib = javaLibs
    jlibdir = fullfile(libJava, jlib);
    d = dir(fullfile(jlibdir, '*.jar'));
    for jar = {d.name}
      javaaddpath(fullfile(jlibdir, jar));
    end
  end
end

% Matlab library dependencies
libMatlab = fullfile(distroot, 'lib', 'matlab');
if isfolder(libMatlab)
  mLibs = readdir(libMatlab);
  for mlib = mLibs
    mlibdir = fullfile(libMatlab, mlib);
    % There's no standard layout for a Matlab project, so we use heuristics to guess
    % where they keep their source files
    candidateSubdirs = ["Mcode" "mcode" "src" "srcfiles"];
    for sub = candidateSubdirs
      if isfolder(fullfile(mlibdir, sub))
        addpath(fullfile(mlibdir, sub))
      end
    end
    d = dir(fullfile(mlibdir, '*.m'));
    if ~isempty(d)
      addpath(mlibdir);
    end
    % TODO: Maybe we should just look at all top-level dirs that aren't `+` package dirs,
    % and add them all if they contain any M-files?
  end
end

% Put any other custom library initialization code here

% Mark library as initialized

mypackage.internal.misc.setpackageappdata('initialized', true);

end

function out = readdir(theDir)
d = dir(theDir);
out = string(setdiff({d.name}, {'.' '..'}));
end
