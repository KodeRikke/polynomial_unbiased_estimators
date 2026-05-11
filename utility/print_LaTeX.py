import subprocess
from pathlib import Path

"""
Utility function to build a LaTeX document from a given LaTeX body string, and compile it to PDF using pdflatex.
The function creates a .tex file with the given output stem (the filename without extension), 
writes the LaTeX content to it, and then runs pdflatex to generate the PDF.
The LaTeX document includes a basic preamble with the amsmath and amssymb packages, 
alongside the package breqn (for splitting equations), and allows for a title to be specified.
Parameters:
- latex_body: The main content of the LaTeX document (as a string).
- output_stem: The file stem for the output .tex and .pdf files (without the .tex extension).
- title: An optional title for the document. If not provided, the output stem will be used as the title.
"""
def build_latex_document(latex_body: str, output_stem: str, *, title: str | None = None):

    # If the folder "reports" does not exist, create it
    reports_folder = Path("reports")
    reports_folder.mkdir(exist_ok=True)
    # Create the .tex file path from the output stem and the reports folder
    tex_path = reports_folder / Path(output_stem).with_suffix(".tex")

    # Note that for using breqn, some dependencies might be off, needed to run:
    # sudo apt install texlive-latex-recommended
    # in terminal. 
    preamble = [
        r"\documentclass{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{breqn}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
    ]

    if not title: 
        title = output_stem # default title is just the output stem (without .tex suffix)

    preamble += [
        rf"\section*{{{title}}}"
    ]

    ending = [
        r"\end{document}"
    ]

    full_tex = "\n".join(preamble + [latex_body] + ending)
    tex_path.write_text(full_tex, encoding="utf-8")

    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], 
                   cwd=tex_path.parent, 
                   text=True,
                   capture_output=True
               )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("pdflatex failed; see log above for details.")