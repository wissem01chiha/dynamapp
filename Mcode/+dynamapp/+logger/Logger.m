classdef Logger
  %LOGGER Main entry point through which logging happens
  %
  % The Logger class provides method calls for performing logging, and the ability
  % to look up loggers by name. This is the main entry point through which all
  % mypackage.logger logging happens.
  %
  % Usually you don't need to interact with this class directly, but can just call
  % one of the error(), warn(), info(), debug(), or trace() functions in the logger
  % namespace. Those will log messages using the calling class's name as the name
  % of the logger. Also, don't call the constructor for this class. Use the static
  % getLogger() method instead.
  %
  % Use this class directly if you want to customize the names of the loggers to
  % which logging is directed.
  %
  % Each of the logging methods - error(), warn(), info(), debug(), and
  % trace() - takes a sprintf()-style signature, with a format string as
  % the first argument, and substitution values as the remaining
  % arguments.
  %    mypackage.logger.info(format, varargin)
  % You can also insert an MException object at the beginning of the
  % argument list to have its message and stack trace included in the log
  % message.
  %    mypackage.logger.warn(exception, format, varargin)
  %
  % See also:
  % mypackage.logger.error
  % mypackage.logger.warn
  % mypackage.logger.info
  % mypackage.logger.debug
  % mypackage.logger.trace
  %
  % Examples:
  %
  % log = mypackage.logger.Logger.getLogger('foo.bar.FooBar');
  % log.info('Hello, world! Running on Matlab %s', version);
  %
  % try
  %     some_operation_that_could_go_wrong();
  % catch err
  %     log.warn(err, 'Caught exception during processing')
  % end
  
  properties (SetAccess = private)
    % The underlying SLF4J Logger object
    jLogger
  end
  
  properties (Dependent = true)
    % The name of this logger
    name
    % A list of the levels enabled on this logger
    enabledLevels
  end
  
  
  methods (Static)
    function out = getLogger(identifier)
      % Gets the named Logger
      jLogger = org.slf4j.LoggerFactory.getLogger(identifier);
      out = mypackage.logger.Logger(jLogger);
    end
  end
  
  methods
    function this = Logger(jLogger)
      %LOGGER Build a new logger object around an SLF4J Logger object.
      %
      % Generally, you shouldn't call this. Use logger.Logger.getLogger() instead.
      mustBeA(jLogger, 'org.slf4j.Logger');
      this.jLogger = jLogger;
    end
    
    function disp(this)
      %DISP Custom object display.
      disp(dispstr(this));
    end
    
    function out = dispstr(this)
      %DISPSTR Custom object display string.
      if isscalar(this)
        strs = dispstrs(this);
        out = strs{1};
      else
        out = sprintf('%s %s', size2str(size(this)), class(this));
      end
    end
    
    function out = dispstrs(this)
      %DISPSTRS Custom object display strings.
      out = cell(size(this));
      for i = 1:numel(this)
        out{i} = sprintf('Logger: %s (%s)', this(i).name, ...
          strjoin(this(i).enabledLevels, ', '));
      end
    end
    
    function error(this, msg, varargin)
      % Log a message at the ERROR level.
      if ~this.jLogger.isErrorEnabled()
        return
      end
      msgStr = formatMessage(msg, varargin{:});
      this.jLogger.error(msgStr);
    end
    
    function warn(this, msg, varargin)
      % Log a message at the WARN level.
      if ~this.jLogger.isWarnEnabled()
        return
      end
      msgStr = formatMessage(msg, varargin{:});
      this.jLogger.warn(msgStr);
    end
    
    function info(this, msg, varargin)
      % Log a message at the INFO level.
      if ~this.jLogger.isInfoEnabled()
        return
      end
      msgStr = formatMessage(msg, varargin{:});
      this.jLogger.info(msgStr);
    end
    
    function debug(this, msg, varargin)
      % Log a message at the DEBUG level.
      if ~this.jLogger.isDebugEnabled()
        return
      end
      msgStr = formatMessage(msg, varargin{:});
      this.jLogger.debug(msgStr);
    end
    
    function trace(this, msg, varargin)
      % Log a message at the TRACE level.
      if ~this.jLogger.isTraceEnabled()
        return
      end
      msgStr = formatMessage(msg, varargin{:});
      this.jLogger.trace(msgStr);
    end
    
    function errorj(this, msg, varargin)
      % Log a message at the ERROR level, using SLFJ formatting.
      if ~this.jLogger.isErrorEnabled()
        return
      end
      this.jLogger.error(msg, varargin{:});
    end
    
    function warnj(this, msg, varargin)
      % Log a message at the WARN level, using SLFJ formatting.
      if ~this.jLogger.isWarnEnabled()
        return
      end
      this.jLogger.warn(msg, varargin{:});
    end
    
    function infoj(this, msg, varargin)
      % Log a message at the INFO level, using SLFJ formatting.
      if ~this.jLogger.isInfoEnabled()
        return
      end
      this.jLogger.info(msg, varargin{:});
    end
    
    function debugj(this, msg, varargin)
      % Log a message at the DEBUG level, using SLFJ formatting.
      if ~this.jLogger.isDebugEnabled()
        return
      end
      this.jLogger.debug(msg, varargin{:});
    end
    
    function tracej(this, msg, varargin)
      % Log a message at the TRACE level, using SLFJ formatting.
      if ~this.jLogger.isTraceEnabled()
        return
      end
      this.jLogger.trace(msg, varargin{:});
    end
    
    function out = isErrorEnabled(this)
      % True if ERROR level logging is enabled for this logger.
      out = this.jLogger.isErrorEnabled;
    end
    
    function out = isWarnEnabled(this)
      % True if WARN level logging is enabled for this logger.
      out = this.jLogger.isWarnEnabled;
    end
    
    function out = isInfoEnabled(this)
      % True if INFO level logging is enabled for this logger.
      out = this.jLogger.isInfoEnabled;
    end
    
    function out = isDebugEnabled(this)
      % True if DEBUG level logging is enabled for this logger.
      out = this.jLogger.isDebugEnabled;
    end
    
    function out = isTraceEnabled(this)
      % True if TRACE level logging is enabled for this logger.
      out = this.jLogger.isTraceEnabled;
    end
    
    function out = listEnabledLevels(this)
      % List the levels that are enabled for this logger.
      out = {};
      if this.isErrorEnabled
        out{end+1} = 'error';
      end
      if this.isWarnEnabled
        out{end+1} = 'warn';
      end
      if this.isInfoEnabled
        out{end+1} = 'info';
      end
      if this.isDebugEnabled
        out{end+1} = 'debug';
      end
      if this.isTraceEnabled
        out{end+1} = 'trace';
      end
    end
    
    function out = get.enabledLevels(this)
      out = this.listEnabledLevels;
    end
    
    function out = get.name(this)
      out = char(this.jLogger.getName());
    end
  end
  
end

function out = formatMessage(format, varargin)
args = varargin;
exceptionStr = [];
if isa(format, 'MException')
  exception = format;
  if isempty(args)
    format = '';
  else
    format = args{1};
    args = args(2:end);
  end
  exceptionStr = getReport(exception, 'extended', 'hyperlinks','off');
  % Remove blank lines for more compact, readable (imho) logs
  exceptionStr = strrep(exceptionStr, sprintf('\n\n'), newline);
end
for i = 1:numel(args)
  if isobject(args{i})
    if isa(args{i}, 'string') && isscalar(args{i})
      if ismissing(args{i})
        args{i} = '<missing>';
      else
        args{i} = char(args{i});
      end
    else
      % General case
      args{i} = dispstr(args{i});
    end
  end
end
out = sprintf(format, args{:});
if ~isempty(exceptionStr)
  out = sprintf('%s\n%s', out, exceptionStr);
end
end
