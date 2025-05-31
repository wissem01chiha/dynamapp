function rmrf(files)
% Recursively delete files and directories
%
% rmrf(files)
%
% files is a list of files and directories to delete. They may contain "*"
% wildcards.
%
% Nonexistent directories and non-matching patterns are silently ignored.
arguments
  files string
end

%#ok<*STRNU>

RAII.warn = withwarnoff('MATLAB:DELETE:FileNotFound');

% Sigh. We have to glob out the files ourselves because Matlab doesn't have a
% delete operation that will work on both files and directories.
for iGlob = 1:numel(files)
  glob = files(iGlob);
  if glob.contains("*")
    [~,details] = mypackage.internal.util.dir2(files(iGlob));
    paths = details.path;
    for file = paths'
      rmrf_one_file(file);
    end
  else
    file = glob;
    rmrf_one_file(file);
  end
end

end

function rmrf_one_file(file)
if isfile(file)
  delete(file);
elseif isfolder(file)
  mypackage.internal.util.rmdir2(file, 's');
else
  % Ignore nonexistent files.
end
end
