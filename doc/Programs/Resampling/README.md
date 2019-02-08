# ResamplingAnalysisScripts

## Sample Scripts for data Analysis
So far this is a simple python script (should be made parallel...) to perform resampling of a data set. Methods used are __Bootstrapping__, __Jackknife__ and __Blocking__.

## Usage
Simply run `python analysis.py FILENAME.xxx [NLINES]`

Where `FILENAME` is expected to have a 3 charachter extension `NLINES` (optional) is the number of lines in the file to read and process (default is the whole file, but it gets very slow above 2-3 hundred thousand entries)

Ouput is located into the `FILENAME/` folder.

If more than 10⁵ lines are specified the autocorrelation function won't be computed, as it would take too long.

The `gaussian.dat` dataset has been generated with numpy, as a proof of concept. It represents a normally distributed set of 5x10⁵ elements with `std = 0.05`. One will notice that the estimate on the error of the central value is greatly improved by all resampling methods.

