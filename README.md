# CLIML

CLIML - an example (cli) python package for machine learning


## Intro

It is usual that ML projects start from jupyter notebook, but the path of
transforming them into a standalone applications is not clear.

This repo contains an example of a callable machine learning python module
that can be executed from the command line.

## Make it run

* Clone the repo
* Install the package `pip install .`

* Generate some sample data `climl datagen > iris.csv`
* Train the model `climl train iris.csv model.pkl`

* Generate inference data `climl datagen --only-inference > iris2.csv`
* Do inference `climl predict model.pkl iris2.csv`

## Make it run with pipes

The idea of this packages is to showcase machine learning pipelines using the
command line

`climl datagen | climl train > model.pkl`

Or even more ambitious using file descriptors, wich will allow you to execute
everythin in one line

`climl predict <(climl datagen | climl train) <(climl datagen --only-inference)`

## Planned Improvements

- Include tests
- Add CI/CD
- Generate a Dockerfile
