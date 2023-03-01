# DeepExperiment

If you want to understand the basic functionality of the package, navigate to the [experiments/Helwak2013_miRBind_interpretation_showcase.ipynb](experiments/Helwak2013_miRBind_interpretation_showcase.ipynb).

## Usage 

You can install this package using pip and github repository as follows:

```bash
pip install git+https://github.com/katarinagresova/DeepExperiment
```

## Local development

If you want to run experiments from this repository or contribute to the package, use following commands to clone the repository and install the package into virtual environment.

```bash
git clone git@github.com:katarinagresova/DeepExperiment.git
cd DeepExperiment

virtualenv venv --python=python3.8
source venv/bin/activate

pip install -e .
```


## Citing Deep Experiment

If you use Deep Experiment in your research, please cite it as follows.

### Text

Grešová, K.; Vaculík, O.; Alexiou, P. Using Attribution Sequence Alignment to Interpret Deep Learning Models for miRNA Binding Site Prediction. Biology 2023, 12, 369. https://doi.org/10.3390/biology12030369

### BibTeX

```bib
@article{gresova2023using,
  title={Using Attribution Sequence Alignment to Interpret Deep Learning Models for miRNA Binding Site Prediction},
  author={Gre{\v{s}}ov{\'a}, Katar{\'\i}na and Vacul{\'\i}k, Ond{\v{r}}ej and Alexiou, Panagiotis},
  journal={Biology},
  year={2023},
  publisher={MDPI},
}
