function debug(msg, varargin)
% Log a DEBUG level message from caller, with printf style formatting.
%
% logger.debug(msg, varargin)
% logger.debug(exception, msg, varargin)
%
% This accepts a message with printf style formatting, using '%...' formatting
% controls as placeholders.
%
% Examples:
%
% logger.debug('Some message. value1=%s value2=%d', 'foo', 42);

loggerCallImpl('debug', msg, varargin);

end