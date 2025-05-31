function loggerCallImpl(logLevel, msg, args, form)
%LOGGERCALLIMPL Implementation for the top-level logger functions
%
if nargin < 3 || isempty(args);  args = {};  end
if nargin < 4 || isempty(form);  form = 'm'; end

% Can't use a regular dbstack call here because the stack info doesn't include
% package names. Resort to parsing the human-readable formatted stack trace.
strStack = evalc('dbstack(2)');
stackLine = regexprep(strStack, '\n.+', '');
% BUG: Caller name detection doesn't work right for anonymous functions.
% This pattern isn't quite right. But truly parsing the HTML would be expensive.
caller = regexp(stackLine, '>([^<>]+)<', 'once', 'tokens');
if isempty(caller)
    % This will happen if you call the log functions interactively
    % Use a generic logger name
    callerId = 'base';
else
    callerEl = caller{1};
    % Could be 'package.class/method', 'package.function', or
    % 'package.class.staticmethod'. (I think, based on R2016b output.)
    if contains(callerEl, '/')
        ixSlash = find(callerEl == '/', 1, 'last');
        callerId = callerEl(1:ixSlash-1);
    else
        % Static method or function
        ixDot = find(callerEl == '.', 1, 'last');
        if isempty(ixDot)
            callerId = callerEl;
        else
            prefix = callerEl(1:ixDot-1);
            maybeClass = meta.class.fromName(prefix);
            if isempty(maybeClass)
                callerId = callerEl;
            else
                callerId = prefix;
            end
        end
    end
end

loggerObj = mypackage.logger.Logger.getLogger(callerId);

switch form
    case 'm'
        switch logLevel
            case 'error'
                loggerObj.error(msg, args{:});
                return
            case 'warn'
                loggerObj.warn(msg, args{:});
                return
            case 'info'
                loggerObj.info(msg, args{:});
                return
            case 'debug'
                loggerObj.debug(msg, args{:});
                return
            case 'trace'
                loggerObj.trace(msg, args{:});
                return
            otherwise
                error('logger:InvalidInput', 'Invalid logLevel: %s', logLevel);
        end
    case 'j'
        switch logLevel
            case 'error'
                loggerObj.errorj(msg, args{:});
                return
            case 'warn'
                loggerObj.warnj(msg, args{:});
                return
            case 'info'
                loggerObj.infoj(msg, args{:});
                return
            case 'debug'
                loggerObj.debugj(msg, args{:});
                return
            case 'trace'
                loggerObj.tracej(msg, args{:});
                return
            otherwise
                error('logger:InvalidInput', 'Invalid logLevel: %s', logLevel);
        end
end

end