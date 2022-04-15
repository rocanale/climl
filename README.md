# CLIML

CLIML - an example (cli) python package for machine learning


## Intro

It is usual that ML projects start from jupyter notebook, but the path of
transforming them into a standalone applications is not clear.

This repo contains an example of a callable machine learning python module
that can be installed or shipped into containers.

# Make it run

* Clone the repo
* Install the package `pip install .`

* Generate some sample data `climl sample-data > iris.csv`
* Train the model `climl train iris.csv model.pkl`

* Generate inference data `climl sample-data --only-inference > iris2.csv`
* Do inference `climl predict model.pkl iris2.csv`

## Planned Improvements

- Include tests
- Allow input from stdin
- Add CI/CD
- Generate a Dockerfile
