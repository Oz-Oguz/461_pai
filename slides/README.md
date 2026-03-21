# PAI Lecture Slides

Marp-based slide decks for all five PAI modules.

## Files

| File | Module | Slides |
|---|---|---|
| `01_bayesian_networks.md` | Bayesian Networks (Ch. 1) | ~22 |
| `02_hidden_markov_models.md` | Hidden Markov Models (Ch. 2) | ~29 |
| `03_bayesian_linear_regression.md` | Bayesian Linear Regression (Ch. 3) | ~24 |
| `04_kalman_filter.md` | Kalman Filter (Ch. 4) | ~23 |
| `05_gaussian_processes.md` | Gaussian Processes (Ch. 5) | ~24 |

## Rendering

**Prerequisites:** Install Marp CLI once:
```bash
npm install -g @marp-team/marp-cli
```

### Quick Rendering (Using Script)

Render all slides to PDF:
```bash
cd slides
./render.sh
```

Render a specific slide by exact filename:
```bash
./render.sh 02_hidden_markov_models.md
./render.sh 03_bayesian_linear_regression.md
```

Output PDFs are saved to `slides/pdfs/`.

### Manual Rendering

Render to PDF:
```bash
marp slides/02_hidden_markov_models.md --pdf --allow-local-files -o slides/pdfs/02_hmms.pdf
```

Render to HTML (interactive, includes presenter notes):
```bash
marp slides/02_hidden_markov_models.md --html -o slides/pdfs/02_hmms.html
```

Render to PPTX (editable in PowerPoint/Keynote):
```bash
marp slides/02_hidden_markov_models.md --pptx -o slides/pdfs/02_hmms.pptx
```

## Math

All equations use **KaTeX** — the same subset of LaTeX supported by Marp. Rendered natively in PDF/HTML output. For PPTX export, equations are rasterised.

## Customisation

- Colours: edit the `style:` block in the frontmatter of each `.md` file
- Add screenshots from the PAI tool: `![alt text](../path/to/screenshot.png)`
- Speaker notes: add `<!-- note: ... -->` after any slide separator `---`
