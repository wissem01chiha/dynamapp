function mypackage_build_all_mex
  % Builds all the MEX files in __myproject__'s source tree
  %
  % This is for use during development.
  %
  % You should do this at code authoring time, and then check your built MEX files back in
  % to git! This is not part of the project build process.
  %
  % This function builds your mex files with the default mex() options. There's no mechanism
  % to specify other options, either globally or on a per-file basis. We should probably
  % add something to do that.

  coderoot = fullfile(mypackage.globals.distroot, 'Mcode');
  mexSourceFiles = [
    searchFilesRecursively(coderoot, '**/*.cpp')
    searchFilesRecursively(coderoot, '**/*.c')
    searchFilesRecursively(coderoot, '**/*.f')
  ];

  for sourceFile = mexSourceFiles'
    fprintf('Building MEX: %s\n', sourceFile)
    mex(sourceFile)
  end

end

function out = searchFilesRecursively(base, pattern)
  d = dir(fullfile(base, pattern));
  out = fullfile({d.folder}, {d.name})';
end
