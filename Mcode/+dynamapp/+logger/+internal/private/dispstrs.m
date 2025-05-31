function out = dispstrs(x)
%DISPSTRS Display strings for array elements
%
% out = dispstrs(x, options)
%
% DISPSTRS returns a cellstr array containing display strings that represent the
% values in the elements of x. These strings are concise, single-line strings
% suitable for incorporation into multi-element output. If x is a cell, each
% element cell's contents are displayed, instead of each cell.
%
% Unlike DISPSTR, DISPSTRS returns output describing each element of the input
% array individually.
%
% This is used for constructing display output for functions like DISP.
% User-defined objects are expected to override DISPSTRS to produce suitable,
% readable output.
%
% The output is human-consumable text. It does not have to be fully precise, and
% does not have to be parseable back to the original input. Full type
% information will not be inferrable from DISPSTRS output. The primary audience
% for DISPSTRS output is Matlab programmers and advanced users.
%
% The intention is for user-defined classes to override this method, providing
% customized display of their values.
%
% The input x may be a value of any type. The main DISPSTRS implementation has
% support for Matlab built-ins and common types. Other user-defined objects are
% displayed in a generic "m-by-n <class> array" format.
%
% Returns a cellstr the same size as x.
%
% Options:
%   None are currently defined. This argument is reserved for future use.
%
% Examples:
%   dispstrs(magic(3))
%
% See also: DISPSTR

if isempty(x)
	out = reshape({}, size(x));
elseif isnumeric(x)
	out = dispstrsNumeric(x);
elseif iscellstr(x)
	out = x;
elseif isstring(x)
    out = cellstr(x);
elseif iscell(x)
	out = dispstrsGenericDisp(x);
elseif ischar(x)
	% An unfortunate consequence of the typical use of char and dispstrs' contract
	out = num2cell(x);
elseif isa(x, 'tabular')
    out = dispstrsTabular(x);
elseif isa(x, 'datetime')
    out = dispstrsDatetime(x);
elseif isa(x, 'struct')
	out = repmat({'1-by-1 struct'}, size(x));
else
	out = dispstrsGenericDisp(x);
end

out = string(out);

end

function out = dispstrsDatetime(x)
out = cell(size(x));
tfFinite = isfinite(x);
out(tfFinite) = cellstr(datestr(x(tfFinite)));
out(isnat(x)) = {'NaT'};
dnum = datenum(x);
out(isinf(dnum) & dnum > 0) = {'Inf'};
out(isinf(dnum) & dnum < 0) = {'-Inf'};
end

function out = dispstrsNumeric(x)
out = reshape(strtrim(cellstr(num2str(x(:)))), size(x));
end

function out = dispstrsTabular(x)
out = cell(size(x));
for iRow = 1:size(x, 1)
    for iCol = 1:size(x, 2)
        val = x{iRow,iCol};
        if iscell(val)
            val = val{1};
        end
        out{iRow,iCol} = dispstr(val);
    end
end
end

function out = dispstrsGenericDisp(x)
out = cell(size(x));
for i = 1:numel(x)
	if iscell(x)
		xi = x{i}; %#ok<NASGU>
	else
		xi = x(i); %#ok<NASGU>
	end
	str = evalc('disp(xi)');
	str(end) = []; % chomp newline
	out{i} = str;
end
end

