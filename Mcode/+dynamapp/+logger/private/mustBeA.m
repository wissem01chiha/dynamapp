function mustBeA(value, type)
%MUSTBETYPE Validate that an input is of a particular data type
%
% MUSTBETYPE(value, type)
%
% Validates that the input Value is of the specified Type or a
% subtype. If Value is not of Type, an error is raised. If Value is
% of Type, does nothing and returns.
%
% Value is the value to validates the type of. It may be anything. If
% you call it using a variable (as opposed to a longer expression),
% the variable name is included in any error messages.
%
% Type (char) is the name of the type that Value must be. A type
% name may be one of:
%   * A class, such as 'double', 'cell', or 'containers.Map'
%   * One of the special SLF4M pseudotypes:
%       cellstr
%       numeric
%       object
%       any
%
% Note: The cellstr pseudotype is nontrivial to check for, as it
% must call iscellstr() and check all cell contents.

% Avoid infinite recursion
assert(ischar(type), 'mypackage:InvalidInput',...
    'type must be a char, but got a %s', class(type));


% Special pseudotype cases
switch type
    case 'cellstr'
        if iscellstr(value)
            return
        else
            if iscell(value)
                elementTypes = unique(cellfun(@class, value, 'UniformOutput',false));
                typeDescription = sprintf('cell containing %s', strjoin(elementTypes, ' and '));
            else
                typeDescription = class(value);
            end
            reportBadValue(inputname(1), type, typeDescription);
        end
    case 'numeric'
        if isnumeric(value)
            return
        else
            reportBadValue(inputname(1), 'numeric', class(value));
        end
    case 'object'
        % 'object' means user-defined Matlab objects
        if isobject(value)
            return
        else
            reportBadValue(inputname(1), 'object', class(value));
        end
    case 'any'
        % Always passes: any type is an 'any'
        return
end

% General case
if ~isa(value, type)
    reportBadValue(inputname(1), type, class(value));
end

end


