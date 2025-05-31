function out = parseOpts(opts, defaults)
%PARSEOPTS Parse a Janklab-style "option" argument
%
% out = parseOpts(opts, defaults)
%
% Parses a Janklab-style "option" argument into a struct, optionally applying
% defaults. Generally you will want to supply a defaults with all fields
% present, so you don't have to check for the existence of a field before
% referencing it.
%
% Options and defaults may both be of type:
%   * struct
%   * cellrec
% 
out = parseOptsArg(opts);
if nargin > 1
    defaults = parseOptsArg(defaults);
    out = copyfields(defaults, out);
end
end

function out = parseOptsArg(opts)
if isempty(opts)
    opts = struct;
end
if isstruct(opts)
    out = opts;
elseif iscell(opts)
    opts = cellrec(opts);
    out = cellrec2struct(opts);
else
    error('dispstr:InvalidInput', 'Unsupported input type: %s', class(opts));
end
end