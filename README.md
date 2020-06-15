# variational_inference

This is a project for experimenting with variational inference and design patterns in software development.
The goal is to get familiar with approximate inference and find strategies for improvements in the optimization 
procedure.

## Getting Started

Follow these instructions to get the project running for development purposes.
### Prerequisites

1. Make sure you installed [pyenv](https://github.com/pyenv/pyenv-installer) to manage your python versions.
Add the following lines to your .bashrc or .bash_profile.
    ```
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    ```
   The python version in your terminal is now set via pyenv.
   
2. Install Python 3.7.7 for the project.
    ```
    pyenv install 3.7.7.
    ```
   Switch into the root directory of the repository and type:
    ```
    pyenv local 3.7.7
    ```

3. Install pipenv to manage package dependencies and create a virtual environment with it.
    ```commandline
    pip install pipenv
    pipenv shell
    ```

### Installing

Install all necessary packages with the help of pipenv, by simply executing:
```
pipenv install
```
It reads out package dependencies from the "Pipfile" and gets the correct versions via hash values.

If you want to install a new package install it via ```pipenv install <packageName>```. It than will be tracked
in the "Pipfile".

Visualize package dependencies via ```pipenv graph```.

## Authors

Michael Gebauer <br> 
E-Mail:
[gebauerm23@gmail.com](mailto:gebauerm23@gmail.com)

## Acknowledgments

This project is inspired by papers and repositories mentioned in the references. Feel free to contribute.
If you use this repository please cite with:
```
@misc{var_inf_geb,
  author = {Gebauer, Michael},
  title = {variational_inference - an introductory example},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gebauerm/variational_inference}},
}
```

## References

* Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. “Variational Inference: A Review for Statisticians.” Journal of the American Statistical Association 112.518 (2017)
* Blei, David M., Andrew Y. Ng, and Michael I. Jordan. \"Latent dirichlet allocation.\" Journal of machine Learning research (2003).
* Kingma, Diederik P., and Max Welling. \"Auto-encoding variational bayes.\" arXiv preprint arXiv:1312.6114 (2013).
* https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
* https://zhiyzuo.github.io/VI
* http://retiredparkingguard.com/posts/2019-06-25-variational-inference-part-1-cavi.html#org69a9b59
* https://www.cs.princeton.edu/courses/archive/fall11/cos597C/
* https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html