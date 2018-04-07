### Forecasting the presence and intensity of hostility on Instagram using linguistic and social features

This notebook provides code to reproduce the primary figures and tables in the paper

> Ping Liu, Joshua Guberman, Libby Hemphill, and Aron Culotta. "Forecasting the presence and intensity of hostility on Instagram using linguistic and social features." In *Proceedings of the Twelfth International AAAI Conference on Web and Social Media (ICWSM'18)*

Note that while all data used was publicly available, in order to respect user privacy and Instagram's terms of service, we are unfortunately unable to share publicly the raw data needed to replicate the results in this notebook. 

### Contents

- [Replication.ipynb](Replication.ipynb): Jupyter notebook to run main experiments.
- [u.py](u.py): data analysis code used by [Replication.ipynb](Replication.ipynb)
- \*.pdf: figures written by [Replication.ipynb](Replication.ipynb)
- [requirements.txt](requirements.txt): python library dependencies

### To run using a [virtualenv](https://pypi.python.org/pypi/virtualenv)
To make sure you're using the same version of all dependencies, you can create a virtualenv and install all dependencies listed in [requirements.txt](requirements.txt) prior to running the notebook.

1. `cd icwsm-2018-hostility`  Enter repository directory.
2. `virtualenv icwsm`  Create a new virtual environment in the directory `icwsm`.
3. `source icwsm/bin/activate` Activate the environment.  
4. `pip3 install -r requirements.txt` (or just `pip`)  Install all dependencies.
5. `jupyter notebook Replication.ipynb`  Start notebook

When you're done running the notebook, you can deactivate the virtualenv and remove the virtualenv directory  

6. `deactivate`  
7. `rm -rf icwsm`


### Authors
- [Ping Liu](https://mypages.iit.edu/~pliu19)
- [Josh Guberman](https://josh.guberman.io/)
- [Libby Hemphill](http://www.libbyh.com/)
- [Aron Culotta](cs.iit.edu/~culotta)
