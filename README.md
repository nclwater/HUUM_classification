# README

This project contains a script which classifies a given (sub-daily) time series's diurnal pattern into four main classes. 
These are:

- Two Peaks
- Morning Peak
- Evening Peak
- Long Peak

Plus, strictly speaking, two more:

- Unsuitable: Time series which are, usually due to overall daily demand value, outside plausible bounds.
- Unclassifyable: Time series, which, while not unsuitable, cannot be classified according to the logic implemented.

This script has been developed for use in the Household Utility Usage Model HUUM during parameter adjustment.
It is part of Sven Berendsen's PhD research 2018-2023 at Newcastle University.
The documentation of the logic in classification can be found in the corresponding thesis (not yet published).


## ToDo

- Introduce more classes to cover the currently unclassifyable cases.
- Professionalise Code: Currently the code is "clean prototype" level, i.e. has little in-code documentation (but very sensible function and object names).
- Improve the derivation of the characteristic points according to previously defined ones.
- Remove the vestiges of matplotlib usage.


## Licence
This code is originally (C) Sven Berendsen, 2023.
Published under the terms of the Apache Licence, version 2.0.

