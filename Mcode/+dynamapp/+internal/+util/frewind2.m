function frewind2(fid)
% A version of frewind that errors on failure
mypackage.internal.util.fseek(fid, 0, 'bof');
end