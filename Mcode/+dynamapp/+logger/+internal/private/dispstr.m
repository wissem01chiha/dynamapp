function out = dispstr(x, options)
%DISPSTR Display string for arrays
%
% out = dispstr(x, options)
%
% This returns a one-line string representing the input value, in a format
% suitable for inclusion into multi-element output. The output describes the
% entire input array in a single string (as opposed to dumping all its
% elements.)
%
% The intention is for user-defined classes to override this method, providing
% customized display of their values.
%
% The input x may be a value of any type. The main DISPSTR implementation has
% support for Matlab built-ins and common types. Other user-defined objects are
% displayed in a generic "m-by-n <class> array" format.
%
% Options may be a struct or an n-by-2 cell array of name/value pairs (names in
% column 1; values in column 2).
%
% Returns a single string as char.
%
% Options:
%   QuoteStrings  - Put scalar strings in quotes.
%
% Examples:
%   dispstr(magic(3))
%
% See also:
% DISPSTRS, SPRINTFD

if nargin < 2;  options = [];  end
options = parseOpts(options, {'QuoteStrings',false});

if ~ismatrix(x)
    out = sprintf('%s %s', mypackage.logger.internal.Dispstr.size2str(size(x)), class(x));
elseif isempty(x)
    if ischar(x) && isequal(size(x), [0 0])
        out = '''''';
    elseif isnumeric(x) && isequal(size(x), [0 0])
        out = '[]';
    else
        out = sprintf('Empty %s %s', mypackage.logger.internal.Dispstr.size2str(size(x)), class(x));
    end
elseif isnumeric(x)
    if isscalar(x)
        out = num2str(x);
    else
        strs = strtrim(cellstr(num2str(x(:))));
        strs = reshape(strs, size(x));
        out = formatArrayOfStrings(strs);
    end
elseif ischar(x)
    if isrow(x)
        if options.QuoteStrings
            out = ['''' x ''''];
        else
            out = x;
        end
    else
        strs = strcat({''''}, num2cell(x,2), {''''});
        out = formatArrayOfStrings(strs);
    end
elseif iscell(x)
    if iscellstr(x)
        strs = strcat('''', x, '''');
    else
        strs = cellfun(@dispstr, x, 'UniformOutput',false);
    end
    out = formatArrayOfStrings(strs, {'{','}'});
elseif isstring(x)
    if options.QuoteStrings
        strs = strcat('"', cellstr(x), '"');
    else
        strs = cellstr(x);
    end
    out = formatArrayOfStrings(strs, {'[',']'});
elseif isa(x, 'datetime') && isscalar(x)
    if isnat(x)
        out = 'NaT';
    else
        out = char(x);
    end
elseif isscalar(x) && (isa(x, 'duration') || isa(x, 'calendarDuration'))
    out = char(x);
elseif isscalar(x) && iscategorical(x)
    out = char(x);
else
    out = sprintf('%s %s', mypackage.logger.internal.Dispstr.size2str(size(x)), class(x));
end

out = string(out);

end

function out = formatArrayOfStrings(strs, brackets)
if nargin < 2 || isempty(brackets);  brackets = { '[' ']' }; end
rowStrs = cell(size(strs,1), 1);
for iRow = 1:size(strs,1)
    rowStrs{iRow} = strjoin(strs(iRow,:), ' ');
end
out = [brackets{1} strjoin(rowStrs, '; ') brackets{2}];
end

