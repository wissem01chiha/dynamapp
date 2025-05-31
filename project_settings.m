function out = project_settings
% Edit these variables to for your project
%
% This file is only used during project initialization. You can throw it away after
% that, but I'd keep it around anyway, just in case.

% The name of your project and its GitHub repo. Capitalization should match what
% your public "branding" is; this will show up in human-readable documentation.
% No spaces, /, or & allowed!
out.PROJECT = "Rmbd";

% The version of Matlab you're developing against. This will be the version of
% Matlab that the project builds against (on Mac and Windows anyway), and the minimum
% required version declared in the project Toolbox file.
out.PROJECT_MATLAB_VERSION = "R2019b";

% The name of the top-level Matlab package that your project defines and keeps
% its code in. This is the "+<package>" directory that'll be directly under Mcode,
% and is the "namespace" that your project lives in.
% It is conventional for package names to be in all lower case.
% Nothing but letters allowed!
out.PACKAGE = "mycoolpackage";

% Your GitHub user name or organization name that's hosting the project
out.GHUSER = "wissem01chiha";

% If you want to provide a contact email for your project, put it here. Optional.
out.PROJECT_EMAIL = "chihawissem08@gmail.com";

% The site generator tool you want to use for the project documentation.
%
% READ THIS PART!
%
% The default is "gh-pages-raw", even though that's an inferior option, because
% that's the only one that works on Windows. If you're on Mac or Linux (or have
% WSL on Windows), pick one of the other options!
%
% Valid choices are:
%
%   "jekyll"       - Regular (full-power) Jeyll for building local docs
%   "mkdocs"       - mkdocs for building local docs
%   "gh-pages"     - GitHub-Pages-compatible (limited) Jekyll
%   "gh-pages-raw" - GitHub Pages raw Markdown files
%
% If you have a large project, you should stick with "jekyll" or "mkdocs" and put your
% main GitHub Pages website in a separate repo. "gh-pages" is more appropriate
% for small projects.
out.DOCTOOL = "gh-pages-raw";

% Human-readable name of the project's primary author or maintainer
out.PROJECT_AUTHOR = "Your Name Here";

% Everything below here is optional! If you omit it, you'll end up with placeholder text
% in some of your documentation, but the project will still work

% One-sentence summary of the project. No <, >, /, or & characters allowed!
out.PROJECT_SUMMARY = "Short summary of project goes here";

% Multi-sentence project description. No <, >, /, or & characters allowed!
out.PROJECT_DESCRIPTION = "Longer description of project goes here.";

% Home page web site for the project's author
out.AUTHOR_HOMEPAGE = "https://github.com/wissem01chiha" + out.GHUSER;

end