function out = reporoot
% The root dir of the myproject repo
out = string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end