# Assignment 3 Report

The report uses the IEEE conference class in two-column, 10-point format and
matches the Assignment 2 authoring workflow.

From this directory in Ubuntu or WSL:

```bash
make check
make
```

The final submission PDF is written to
`../output/pdf/27284808RW741assignment3.pdf`.

The source deliberately avoids active hyperlinks. The repository address is
printed as plain text in a footnote. The completed report uses the frozen pilot
and final evidence, including paired uncertainty and computational costs.

Run `make figures` to rebuild the report-specific vector figures from the frozen
summary CSV files. The figure builder requires the assignment virtual environment
and the installed LaTeX/newtx toolchain. No training runs are launched or changed.
The original experiment figures remain untouched.

The root README contains the setup and build commands.
