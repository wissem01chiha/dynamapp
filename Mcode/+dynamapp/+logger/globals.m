classdef globals
  % Global library properties and settings for SLF4M
  
  properties (Constant)
    % Path to the root directory of this SLF4M distribution
    distroot = string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
  end
  
  methods (Static)
    
    function out = version
      % The version of the SLF4M library
      %
      % Returns a string.
      persistent val
      if isempty(val)
        versionFile = fullfile(logger.globals.distroot, 'VERSION');
        val = strtrim(mypackage.internal.util.readtext(versionFile));
      end
      out = val;
    end
    
  end
  
end

