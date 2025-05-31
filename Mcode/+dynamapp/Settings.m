classdef Settings < mypackage.internal.MypackageBaseHandle
% Global settings for the rmbd package
%
% Don't use this class directly. If you want to get or set the settings,
% work with the instance of this in the mypackage.globals.settings field.

  properties
  end

  methods (Static=true)

    function out = discover()
      % Discovery of initial values for package settings.
      %
      % This could look at config files, environment variables, Matlab appdata, and
      % so on.
      %
      % This needs to avoid referencing mypackage.globals, to avoid a circular dependency.
      out = mypackage.Settings;
    end

  end

end