function pull_in_homebrew_ruby
% Gets Homebrew's Ruby on to your $PATH in this Matlab session
%
% This searches for Homebrew and modifies your $PATH environment variable
% to get Ruby and its ruby, bundler, and other commands on it. This is
% needed on Mac because when you launch Matlab from the /Applications icon,
% it starts up with a minimal $PATH and does not run your shell startup
% files.
%
% If it doesn't find a Homebrewed Ruby installation, it just does nothing
% and returns silently.
%
% Call this from your `startup.m` if you're a Mac Homebrew user. You'll probably
% need to do this if you're using the Jekyll or GitHub Pages documentation
% builder, because they rely on having a new Ruby and Bundler installed, and
% Homebrew is usually the way people do that on Macs.
%
% To use this, stick this in your startup.m (after getting dev-kit/ on your
% Matlab path):
%
% if ~ispc
%   pull_in_homebrew_ruby
% end

%#ok<*NBRAK>

if ispc
  % fprintf("pull_in_homebrew_ruby: Homebrew doesn't exist on Windows. Not pulling it in.\n");
  return
end

installCandidates = [
  "/usr/local"
  ];
found = [];
for cand = installCandidates'
  if isfolder(cand+"/Cellar") && isfolder(cand+"/bin")
    found = cand;
    break
  end
end

if isempty(found)
  % No Homebrew installation detected
  return
end

p = strsplit(getenv('PATH'), ':');
needBinDirs = [brewPrefix+"/bin", brewPrefix+"/opt/ruby/bin"];
rubyGemsVer = []; %#ok<NASGU>
for minorVer = 5:-1:0
    for patchVer = 10:-1:0
        maybeRubyVer = sprintf("3.%d.%d", minorVer, patchVer);
        if isfolder(brewPrefix+"/lib/ruby/gems/"+maybeRubyVer+"/bin")
            rubyGemsVer = maybeRubyVer;
            needBinDirs(end+1) = brewPrefix+"/lib/ruby/gems/"+rubyGemsVer+"/bin"; %#ok<AGROW>
            break
        end
    end
end
for binDir = needBinDirs
  if ~ismember(binDir, p)
    setenv('PATH', binDir+":"+getenv('PATH'));
  end
end

end
