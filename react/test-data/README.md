# PDF upload fixture

`synthetic-upload.pdf` contains original synthetic prose created for AquiLLM's
upload tests. It replaces the externally authored `cpu-intro-2.pdf` textbook
chapter. Tests override the document title and do not rely on the old content.

The PDF and its generator are project test materials under the project license.
It uses standard PDF Helvetica names without embedding third-party font files.
To regenerate, install reportlab in a development environment and run:

```sh
python react/test-data/generate_upload_fixture.py
```

The generator uses ReportLab's invariant mode for deterministic output.
