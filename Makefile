# This Makefile lets you build the project and its documentation, run tests,
# package it for distribution, and so on.
#
# This is mostly provided just as a convenience for developers who like using 'make'.
# All the actual build logic is in the dev-kit/*.m files, which can be run
# directly, without using 'make'. The exception is 'make java', which must be
# run without Matlab running, because Matlab locks the JAR files it has loaded.
#
# Targets provided:
#
#   make docs     - Build the Markdown-stage doco in docs/ (from docs-src/)
#   make doc      - Build the final static doco into doc/ (from docs/)
#
#   make test     - Run the project Matlab unit tests
#
#   make dist     - Build all the project distribution files
#   make toolbox  - Build the project distribution Matlab Toolbox .mltbx file
#   make zips     - Build the project distribution zip files
#
#   make clean    - Remove derived files
#
#   make java     - Build the project's custom Java code
#   make build    - "Build" (p-code & munge) Mcode source files for distribution
#   make m-doc    - Copy the static documentation into mltbx staging area

.PHONY: test
test:
	./dev-kit/run_matlab "mypackage_make test"

.PHONY: build
build:
	./dev-kit/run_matlab "mypackage_make build"

# Build the programmatically-generated parts of the _source_ files for the doco
.PHONY: docs
docs:
	./dev-kit/run_matlab "mypackage_make docs"

# Build the actual output documents
.PHONY: doc
doc:
	./dev-kit/run_matlab "mypackage_make doc"

.PHONY: m-doc
m-doc:
	./dev-kit/run_matlab "mypackage_make m-doc"

.PHONY: toolbox
toolbox: m-doc
	./dev-kit/run_matlab "mypackage_make toolbox"

.PHONY: zips
zips:
	./dev-kit/run_matlab "mypackage_make zips"

.PHONY: dist
dist:
	./dev-kit/run_matlab "mypackage_make dist"

# TODO: Port this to M-code. This is hard because the .jar cannot be copied in to place
# in lib while Matlab is running, because it locks loaded .jar files (at least on Windows).
.PHONY: java
java:
	cd src/java/myproject-java; mvn package
	mkdir -p lib/java/myproject-java
	cp src/java/myproject-java/target/*.jar lib/java/myproject-java

.PHONY: clean
clean:
	./dev-kit/run_matlab "mypackage_make clean"

# Run this _after_ initialization if you want to throw away some nonessential
# features to make your repo layout simpler.
.PHONY: simplify
simplify:
	./dev-kit/run_matlab "mypackage_make simplify"
