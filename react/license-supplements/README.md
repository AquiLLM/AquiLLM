# Upstream license supplements

These exact npm package versions omit their monorepo's license file from the
published tarball. The files here restore the unmodified upstream license text.
The Vite notices plugin matches the package name **and version**; an upgrade
without a packaged license requires renewed verification, not a silent fallback.

| Package | Verified upstream source (npm registry gitHead) |
| --- | --- |
| remark-math 6.0.0 | https://github.com/remarkjs/remark-math/blob/d5d0660b150810a535bbb07eac6cc96a4510aa24/license |
| rehype-katex 7.0.1 | https://github.com/remarkjs/remark-math/blob/88a9497e1ede93b958237c85edbf5651faeca7af/license |

These are MIT licenses owned by the named upstream copyright holders. Do not
replace them with the AquiLLM project license or remove their attribution.

`katex@0.16.45.txt` additionally preserves the font-specific copyright notices
and reserved font names from the Windows nameID 13 fields of all 20 TTFs shipped
in `katex@0.16.45/dist/fonts`, together with full SIL OFL 1.1 terms. The root
KaTeX MIT license does not replace these font notices. The supplement is
version-bound and includes the Design Science / Khan Academy attribution.
