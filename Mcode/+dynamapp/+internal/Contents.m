% The +internal package contains code that is for the internal use of this
% program. It is not part of its public API, and may change at any time.
%
% The +internal package exists because it is not practical to protect 
% internal-use code with "Access=private" access controls; that makes it hard to
% break code up into multiple classes, makes it hard to test, and can't be
% applied to functions outside classes anyway.