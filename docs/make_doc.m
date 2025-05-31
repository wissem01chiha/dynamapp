function make_doc
% Build these doc sources into the final doc/ directory

%#ok<*STRNU>

import mypackage.internal.util.*

RAII.cd = withcd(fileparts(mfilename('fullpath')));
rmdir2('../doc', 's');
mkdir2('../doc');
copyfile2('*.*', '../doc');

end

