# RMBD Release Checklist

## Doing a release with the script

The `dev-kit/make_release` script can take care of most of the release work for you. To use it:

* Update `CHANGES.md` with the release date for your version.
* Commit the changes.
* Run `./dev-kit/make_release <version>`.
* Go to your GitHub repo site and draft the release.

## Doing a release manually

* Run all the tests.
  * `make test`, duh.
  * Wouldn't hurt to do `make clean && git status && make test`/manual-cleanup, just to be sure.
* Update and double-check the version number and date in `VERSION`.
* Update the installation instructions in README to use the upcoming new release tarball URL.
  * Format is: `https://github.com/__myghuser__/__myproject__/releases/download/v<version>/__myproject__-<version>.tar.gz`
* Regenerate the doco.
  * `make doc`
* Commit all the files changed by the above steps.
  * Use form: `git commit -a -m 'Cut release v<version>'`
* Make sure your repo is clean: `git status` should show no local changes.
* Build it!
  * Run `make dist`.
  * Be sure to do this first, before creating the tag.
* Create a git tag and push it and the changes to GitHub.
  * `git tag v<version>`
  * `git push; git push --tags`
* Create a new GitHub release from the tag.
  * Just use `<version>` as the name for the release.
  * Upload the dist tarball as a file for the release.
* Open development for next version.
  * Update version number in `VERSION` to have a "+" suffix.
  * Rebuild the doco.
    * `(cd doc; make maintainer-clean; make all)`
  * `git commit -a -m 'Open development for v<version>'; git push`

* If there were any problems following these instructions exactly as written, report it as a bug.
