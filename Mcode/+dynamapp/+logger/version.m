function out = version
% VERSION Get version info for SLF4M
%
% logger.version
% v = logger.version
%
% Gets version info for the SLF4M library.
%
% If return value is not captured, displays version info for SLF4M and related
% libraries to the console.
%
% If return value is captured, returns the version of the SLF4M library as
% a char vector.

slf4mVersion = getSlf4mVersion;

if nargout > 0
    out = slf4mVersion;
    return
else
    versions.SLF4M = slf4mVersion;
    versions.SLF4J = '?';
    versions.log4j = '?';
    % TODO: Figure out how to extract the version info from the JARs
    % This probably can't be done without comparing JAR file checksums to
    % Maven Central, which is slow, so maybe this isn't worth doing.
    jars = javaclasspath('-static');
    for i = 1:numel(jars)
        jarFile = jars{i};
        if endsWith(jarFile, 'log4j.jar')
            %jar = java.util.jar.JarFile(jarFile);
            %manifest = jar.getManifest;
        elseif endsWith(jarFile, 'slf4j-api.jar')
        end
    end
    libnames = fieldnames(versions);
    for i = 1:numel(libnames)
        fprintf('%s version %s\n', libnames{i}, versions.(libnames{i}));
    end
end

end

function out = getSlf4mVersion
thisFile = mfilename('fullpath');
rootDir = fileparts(fileparts(fileparts(thisFile)));
versionFile = fullfile(rootDir, 'VERSION');
txt = fileread(versionFile);
out = regexprep(txt, '\r?\n', '');
end
