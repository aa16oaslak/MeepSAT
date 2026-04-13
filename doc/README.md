The following directory tree contains the documentation contents of MeepSAT using `mkdocs`.

### How to use this documentation directory?

To build and visualize the HTML documentation locally using the mkdocs package (useful for verifying changes on your local machine before committing), first install mkdocs in a conda environment.

```bash
pip install mkdocs
```

Also install the following packages as well:
```bash
pip install pymdown-extensions
```

To preview your documentation with live reload enabled, go to the docs directory `../doc/meepsat_docs/`:

```bash
python -m mkdocs serve --livereload
```