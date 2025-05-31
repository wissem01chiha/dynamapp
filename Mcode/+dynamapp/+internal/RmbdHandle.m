classdef RmbdBaseHandle < handle
  % This class is a trick to support automatic library initialization
  %
  % To use it, have all your classes that depend on the library being
  % initialized inherit from this or MypackageBase.
  
  properties (Constant, Hidden)
    initializer = mypackage.internal.LibraryInitializer;
  end
  
end

