# VeryDiff

VeryDiff is a tool for the equivalence verification of neural networks (NNs).
Given two NNs and a specification of an input region, it can answer the following question:

**Do the two NNs behave *equivalently* on the given input space?**

So far VeryDiff supports three different kinds of equivalence that can be checked for two NNs:
- $\varepsilon$-equivalence: The numerical outputs of the two NNs vary at most by $\varepsilon$ w.r.t. the $L_\infty$-norm
- Top-1 equivalence: The two NNs provide the same classification outputs
- $\delta$-Top-1 equivalence: If for some input the first NN provides classification $c$ with confidence larger $\delta > 0.5$, then the second NN also yields classification $c$.


The guarantees provided by VeryDiff are **sound**, i.e. if VeryDiff says two NNs are equivalent, they are provably so.
On the other hand, in some cases VeryDiff cannot be complete.
However, VeryDiff always tries to find counterexamples and outputs them if found.

## Installation
This software requires Julia.
We recommend Julia 1.10, but 1.11 may also work.

The software can be installed as follows:

```
git clone https://github.com/samysweb/VeryDiff
cd VeryDiff
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/samysweb/VNNLib.jl", rev="c6bf990"); Pkg.pin("VNNLib"); Pkg.resolve()'
# The next line builds VeryDiff and may take approx. 20 minutes
julia --project=. -e 'using Pkg; Pkg.build()'
```

Regarding the last line: It is currently necessary to explicitly install VNNLIB.jl from Github, as this package is not (yet) available in the Julia registry.

By default VeryDiff requires Gurobi.  
If you do not want to use Gurobi and/or do not have a Gurobi license, set [this line](https://github.com/samysweb/VeryDiff/blob/95811c272307a511130940fa9deaf7eb2c7a5787/src/VeryDiff.jl#L12) to `false`.  
(This configuration option will be improved upon in future versions).

If you have any issues running VeryDiff, please open an issue and/or contact us -- we are happy to help.

## Running the tool

Once the build command has terminated, the binary of the tool can then be found in `./VeryDiff/deps/VeryDiff/bin/VeryDiff`

Manual:
```
usage: <PROGRAM> [--epsilon EPSILON] [--top-1]
                 [--top-1-delta TOP-1-DELTA] [--timeout TIMEOUT]
                 [--naive] [-h] net1 net2 spec

positional arguments:
  net1                  First NN file (ONNX)
  net2                  Second NN file (ONNX)
  spec                  Input specification file (VNNLIB)

optional arguments:
  --epsilon EPSILON     Verify Epsilon Equivalence; provides the
                        epsilon value (type: Float64, default: -Inf)
  --top-1               Verify Top-1 Equivalence
  --top-1-delta TOP-1-DELTA
                        Verify δ-Top-1 Equivalence; provides the delta
                        value (type: Float64, default: -Inf)
  --timeout TIMEOUT     Timeout for verification (type: Int64,
                        default: 0)
  --naive               Use naive verification (without differential
                        verification)
  -h, --help            show this help message and exit
```