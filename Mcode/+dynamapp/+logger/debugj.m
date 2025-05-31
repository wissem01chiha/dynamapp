function debugj(msg, varargin)
% Log a DEBUG level message from caller, using SLF4J style formatting.
%
% logger.debug(msg, varargin)
%
% This accepts a message with SLF4J style formatting, using '{}' as placeholders for
% values to be interpolated into the message.
%
% Examples:
%
% logger.debugj('Some message. value1={} value2={}', 'foo', 42);

loggerCallImpl('debug', msg, varargin, 'j');

end