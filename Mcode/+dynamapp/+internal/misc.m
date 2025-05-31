classdef misc
  % Miscellaneous utilities
  %
  % The ones in this class are ones you typically wouldn't want to import, and are mostly
  % for the +internal package's own internal use.
  
  methods (Static=true)
    
    function mkdir(dir)
      [ok,msg] = mkdir(dir);
      if ~ok
        error('Failed creating directory "%s": %s', dir, msg);
      end
    end
    
    function out = getpackageappdata(key)
      ad = getappdata(0, 'mypackage');
      if isempty(ad)
        ad = struct;
      end
      if isfield(ad, key)
        out = ad.(key);
      else
        out = [];
      end
    end
    
    function setpackageappdata(key, value)
      ad = getappdata(0, 'mypackage');
      if isempty(ad)
        ad = struct;
      end
      ad.(key) = value;
      setappdata(0, 'mypackage', ad);
    end
    
  end
  
end