""" imports """
import sys
import os
import signal
import time
from math import sqrt, log
from tracemalloc import start

from legion.helper import parse_arguments, write_metadata, interrupt, write_testcase, int_to_bytes, gcov, try_remove
from legion.execution import compile_symcc, execute_with_input, trace_from_file, zip_files, run
from legion.tree import *


""" constants """
VERSION = "testcomp2023"
RHO = 1
BITS = 64
INPUTS = set()
KNOWN = set()


""" methods """
def uct(w, n, N):
    """Compute and return the uct score (higher is better)"""

    if not n:
        return inf
    else:
        exploit = w / n
        explore = RHO * sqrt(2 * log(N) / n)
        return exploit + explore


def naive(solver, target):
    """Compute and return the sample in a naive way"""

    assert target
    assert type(target) == list

    if len(target) == 1:
        target = target[0]
    else:
        target = z3.Concat(list(reversed(target)))

    n = target.size()

    delta = z3.BitVec("delta", n)
    result = z3.BitVec("result", n)

    solver.add(result == target)
    solver.minimize(delta)

    while True:
        guess = z3.BitVecVal(random.getrandbits(n), n)

        solver.push()
        solver.add(result ^ delta == guess)

        """idea to integrate the KNOWN constant"""
        # for known in KNOWN:
        #     if result.size() == known.size():
        #         solver.add(result != known)

        if solver.check() != z3.sat:
            break

        model = solver.model()
        value = model[result]

        sample = int_to_bytes(value.as_long(), n // 8)

        solver.pop()

        KNOWN.add(value)
        INPUTS.add(sample)
        yield sample



""" main program """
if __name__ == "__main__":

    """global variables"""
    start = time.time()
    iteration = 0
    ntracefile = 0
    ntestcases = 0  
    last = None
    reach_error = False


    # print version and terminate
    if len(sys.argv) == 2 and (sys.argv[1] == "-v" or sys.argv[1] == "--version"):
        print(VERSION)
        sys.exit(0)

    # set maximum depth of the Python interpreter stack to 1000 * 1000 
    sys.setrecursionlimit(1000 * 1000)

    args = parse_arguments()

    # handle comitted arguments
    if args.rho:
        RHO = args.rho

    source = args.file
    is_c = source[-2:] == ".c" or source[-2:] == ".i"

    if args.dfs:
        dfs = True
    else:
        dfs = False

    if args.m32:
        BITS = 32
    elif args.m64:
        BITS = 64

    if is_c:
        binary = args.binary if args.binary else source[:-2]
        object = source[:-2]
        compile_symcc(args.library, source, binary, BITS, args.coverage)
    else:
        binary = source
        object = source
        source = binary + ".c"
    
    maxlen = None
    if args.maxlen:
        maxlen = args.maxlen

    finishtime = 900
    if args.finish:
        finishtime = args.finish
    
    k = 1
    if args.kExecutions:
        k = args.kExecutions

    stem = os.path.basename(binary)

    # define the root node
    root = Node([], "", [], [])

    # write metadata.xml to tests/
    print("write metadata")
    write_metadata(source, "tests/" + stem, BITS)


    # write empty test case for testcov
    print("write empty testcase...")
    write_testcase(None, "tests/" + stem, iteration)


    signal.signal(signal.SIGTERM, interrupt)


    """ main loop"""
    while True:
        iteration += 1

        # terminate when max iterations is reached
        if args.iterations and iteration >= args.iterations:
            print("max iterations")
            break

        if args.finish and time.time() > start + finishtime:
            print ("max execution time")
            break

        try:

            """ 1. selection phase """
            if args.verbose:
                print("selecting...")

            node = root.select(dfs)

            # handle is_leaf
            if node.is_leaf:
                continue

            """ 2. sampling phase """
            if args.verbose:
                print("sampling...")

            prefix = node.sample()
            if prefix is None:
                node.propagate(0,1)
                if args.verbose:
                    # sample at this path but no more choices locally (solver returned unsat), continue
                    print("e", node.path.ljust(32))
                continue
            else:
                # sample at the path, the input prefix should be shown on the right
                if not args.quiet:
                    print("?", node.path.ljust(32), "input: " + prefix.hex())

            # adapt maximum trace length
            if args.adaptive or not args.maxlen:
                if iteration < 100:
                    maxlen = 100
                elif iteration < 1000:
                    maxlen = 1000
                else:
                    maxlen = 10000

            # do we need this????
            if maxlen and args.maxlen and maxlen > args.maxlen:
                maxlen = args.maxlen

            n = 0
            # integrate the option to execute the computed sample k times
            for n in range(k):
                ntracefile += 1

                """ 3. execution phase """
                if args.verbose:
                    print("executing", binary)

                if args.quiet:
                    traceid = "trace"
                else:
                    traceid = ntracefile

                code, outs, errs, symcc_log, verifier_out = execute_with_input(binary, prefix, "traces/" + stem, ntracefile, args.timeout, maxlen)

                # handle code
                if args.verbose:
                    if -31 <= code and code < 0:
                        print("signal: ", signal.Signals(-code).name)
                    elif code != 0:
                        print("return code: ", code)

                # handle outs
                if outs:
                    if args.verbose:
                        print("stdout:")
                    for line in outs:
                        print(line.decode("utf-8").strip())

                # handle errs
                if errs:
                    if args.verbose:
                        print("stderr:")
                        for line in errs:
                            print(line.decode("utf-8").strip())

                """ 4. readout trace from file """
                try:
                    if args.verbose:
                        print("parse trace", symcc_log)

                    is_complete, last, trace, decls = trace_from_file(symcc_log)

                except Exception as e:
                    node.propagate(0, 1)
                    if args.verbose:
                        print("invalid trace", e)
                    continue

                # handle is_complete
                if not is_complete:
                    if args.verbose:
                        print("partial trace: ", last)
                
                # handle trace ????
                #if not trace:
                #    if args.verbose:
                #        print("deterministic")
                #    continue

                
                # compute path
                bits = ["1" if bit else "0" for (_, _, bit, _) in trace]
                if args.verbose:
                    # path returned by binary execution
                    print("<", "".join(bits))

                """ 5. insertion phase """
                added, leaf = root.insert(trace, is_complete, decls)

                # handle trace
                if trace:
                    _, _, path, _ = zip(*trace)
                

                """ 6. propagation phase """
                if added:
                    node.propagate(1, 1)
                else:
                    node.propagate(0, 1)
                
                # handle added
                if added:
                    # write test case 
                    if not args.error or code == 1:
                        if args.verbose:
                            print("write testcase", verifier_out)
                        else:
                            print("write testcase", ntestcases)
                        write_testcase(verifier_out, "tests/" + stem, iteration)
                        ntestcases += 1
                        # new path found and integrated into the tree
                        if not args.quiet:
                            print("+", leaf.path)

                        # handle error case
                        if last == "error" and args.error:
                            print("write testcase", ntestcases, "(error found)")
                            write_testcase(verifier_out, "tests/" + stem, iteration)
                            reach_error = True
                            print("reach_error() detected.")
                            break

                elif not leaf.path.startswith(node.path):
                    # path failed to preserve the prefix chosen (approximate sampler)
                    if not args.quiet:
                        print("!", leaf.path)

            # abort if the error case is detected
            if last == "error" and args.error:
                break

        # handle exceptions
        except KeyboardInterrupt:
            print("keyboard interrupt")
            break
        except StopIteration:
            print("signal interrupt")
            break
        except Exception as e:
            if args.verbose:
                print()
                if not args.quiet:
                    print("current tree")
                    root.pp_legend()
            raise e


    """ print output and finish """
    print("done")
    print()

    if not args.quiet:
        print("final tree")
        root.pp_legend()
        print()

    print("tests: " + str(ntestcases))
    print()

    # compute coverage information
    if args.coverage:
        print("computing coverage")
        stem = os.path.basename(object)
        gcda = stem + ".gcda"
        gcov(gcda)
        try_remove(gcda)
        try_remove("Verifier.gcda")

    # print the error score
    if args.error and reach_error:
        print("score: 1")

    # handle a execution in testcov mode 
    if args.testcov or args.zip:
        suite = "tests/" + stem + ".zip"
        zip_files(suite, ["tests/" + stem])
        print()

        cmd = ["testcov"]

        if args.m32:
            cmd.append("-32")
        else:
            cmd.append("-64")

        cmd.extend(
            [
                "--no-isolation",
                "--no-plots",
                "--timelimit-per-run",
                "3",
                "--test-suite",
                suite,
                source,
            ]
        )

        if args.testcov:
            run(*cmd)
        else:
            print(*cmd)