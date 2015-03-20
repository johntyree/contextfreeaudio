Just a quick prototype.  Not very fast nor stable.  Requires pyMedia library for saving mp3s and pyparsing for file loading.

Features include pseudo-infinite recursion (to Nyquist), ADSR envelopes, external samples.

Currently not very configurable nor documented.

usage: cfa.py sourcefile.txt outfile.mp3

Syntax is very similar to Context Free Art.  Significant differences include use of the pipe character for parallelization, rule declarations without first saying 'rule', parenthesis instead of curly braces, and choice of scaling or incrementing inherited variables.

Configuration variables include
> frequency float
> amplitude float
> harmonic int
> loop time|count
> duration float|int
> fill new|clone|quiet
> a, d, s, r float

Variables are set by
> variable operation value

Where operations is
> = assign
  * scale
> + increment