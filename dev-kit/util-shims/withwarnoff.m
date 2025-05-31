function out = withwarnoff(warningId)
% Temporarily disable warnings
arguments
  warningId string
end
out = mypackage.internal.util.withwarnoff(warningId);
end
