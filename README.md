# pypillometry
pupillometry with python


## Setup 

### Using `conda`

~~~
$ conda create -n pypil python=3
$ conda activate pypil
$ conda install anaconda 
$ conda activate pypil
~~~

To enable plotting widgets in jupyter notebook and jupyter lab

~~~
$ conda install ipywidgets nodejs
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
~~~

For `pystan`, I needed to do
~~~
pip install pystan
conda install gcc_linux-64
conda install gxx_linux-64
~~~
otherwise, there were random crashes of the jupyter kernel for some reason.