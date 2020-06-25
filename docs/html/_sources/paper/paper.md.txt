---
title: 'pypillometry: A Python package for pupillometric analyses'
tags:
  - Python
  - pupil
  - pupillometry
  - psychology
  - neuroscience
  - LC-NE
authors:
  - name: Matthias Mittner
    orcid: 0000-0003-0205-7353
    affiliation: 1
affiliations:
 - name: Institute for Psychology, UiT The Arctic University of Norway, Norway
   index: 1
date: 19 May 2020
bibliography: paper.bib

---

# Summary

The size of the human pupil is controlled by pairs of constrictor and dilator muscles that allow its opening (dilation) and closing (constriction) in response to varying lighting conditions [@mathot2018pupillometry]. Importantly, it has long been known that the pupil also reacts to psychological important stimuli [@hess1960pupil] and has been a firmly established tool for studying "mental effort" in the research kit of psychologists for many decades [@laeng2012pupillometry]. More recently, pupil-size has been linked to the norepinephrinergic (NE) system originating from area *locus coeruleus* (LC) in the brainstem [@aston2005integrative], a link that has been substantiated experimentally by direct recordings in the brainstem of monkeys [@joshi2016relationships]. This finding of a correlation between NE activity in the brainstem and pupil-dilation has opened the way for researchers investigating the relationship between the LC-NE system and many cognitive functions, such as cognitive control [@gilzenrat2010pupil] and mind wandering [@mittner2016neural]. Advancing this emerging field requires the decomposition of the pupillometric signal into tonic (baseline) and phasic (response) components that relate to different processing regimes of the LC-NE system. 

The Python package `pypillometry` is a comprehensive library implementing preprocessing, plotting and advanced analysis tools in a coherent and extensible, object-oriented framework. It is oriented towards researchers in psychology and neuroscience that wish to analyze data from pupillometric experiments. `pypillometry` implements an intuitive, pipeline-based processing strategy where an analysis pipeline can be dynamically chained together. All operations and parameters applied to a dataset are stored in its history. This allows (1) a transparent and comprehensive logging of the operations applied for an analysis which is valuable for reproducible analyses, (2) the ability to "roll-back" any changes made to any point in the history of the dataset and (3) to easily generalize a processing pipeline to multiple datasets. The package contains pre-processing facilities implementing algorithms for blink detection and interpolation, filtering and resampling. All parameters are clearly documented, accessible and set to sensible default-parameters. A focus of the package is to provide extensive visualization features in order to facilitate dynamic exploration of the data as well as iterative adjustment of the pre-processing parameters. As the time-series of pupillometric data can be quite long, this requires separation into several plots or dynamically adjustable plot-axes. Both strategies are implemented in this package by allowing interactive plots if run from a Jupyter-Notebook [@kluyver2016jupyter] or storing a multi-page PDF document, allowing both interactive and scripted use. The `pypillometry` package also implements functions for event-related pupil-dilation (ERPD) analyses both at the individual and the group-level. Finally, the package implements novel algorithms for decomposing the pupillometric signal into tonic and phasic components. This approach allows to simultaneously quantify dynamic changes of both baseline and response-strength that can be related to the tonic and phasic processing regimes of the LC-NE system.

`Pypillometry` was already used for the analyses of several pupillometric datasets in our department. Several software packages with similar goals are available in R [e.g., @geller_gazer_2020; @forbes2020]. However, to date, no comprehensive Python-based solution besides `pypillometry` exists. None of the other packages provides facilities to estimate tonic and phasic components of the pupillometric signal. 

# Acknowledgements

I would like to thank the members of my research group for stimulating discussions and critical advice. In particular, I would like to thank Isabel Viola Kreis, Josephine Maria Groot, Gábor Csifcsák and Carole Schmidt for their input. I would also like to thank Bruno Laeng for inspiring discussions.

# References