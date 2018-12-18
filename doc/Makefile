# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pydygp
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Also clean Mac OSX sym link files in the examples directories which can corrupt
# sphinx build, and back up files made by emacs while editing
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf ./source/auto_examples/
	rm -rf ./source/tutorials/lfm_tutorials/
	rm -rf ./source/tutorials/llfm_tutorials/
	rm -rf ./source/tutorials/mlfm_adapgrad/
	rm -rf ./source/tutorials/mlfm_adapgrad_tutorials
	rm -rf ./source/tutorials/mlfm_adapgrad
	rm -rf ./source/tutorials/mlfm_sa_tutorials
	rm -rf ./tutorials/mlfm_sa_tutorials
	rm -rf ./source/generated/
	rm -rf ../tutorials/lfm_tutorials/.#*
	rm -rf ../tutorials/llfm_tutorials/.#*
	rm -rf ../tutorials/mlfm_adapgrad_tutorials/.#*
	rm -rf ./source/*.*~

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
