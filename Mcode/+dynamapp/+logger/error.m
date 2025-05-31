function error(msg, varargin)
% Log an ERROR level message from caller, with printf style formatting.
%
% logger.error(msg, varargin)
% logger.error(exception, msg, varargin)
%
% This accepts a message with printf style formatting, using '%...' formatting
% controls as placeholders.
%
% Examples:
%
% logger.error('Some message. value1=%s value2=%d', 'foo', 42);

loggerCallImpl('error', msg, varargin);

end