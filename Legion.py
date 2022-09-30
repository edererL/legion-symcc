#!/usr/bin/python3

# Handle the required imports
import os
import random
import sys
import signal

from math import sqrt, log, inf

from legion.helper import *
from legion.execution import *
from legion.tree import *


# constants
BFS = True
RHO = 1
INPUTS = set()
KNOWN = set()
VERSION = "testcomp2022"
BITS = 64


def interrupt(number, frame):
    """Interrupt and stop interation"""
    print("received SIGTERM")
    raise StopIteration()


# higher is better
def uct(w, n, N):
    """Uct score function:
       Computes and returns the uct score"""
    if not n:
        return inf
    else:
        exploit = w / n
        explore = RHO * sqrt(2 * log(N) / n)
        return exploit + explore


def naive(solver, target):
    """Computes a sample in a naive way"""
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
        # print('---------------------------')
        guess = z3.BitVecVal(random.getrandbits(n), n)

        solver.push()
        solver.add(result ^ delta == guess)

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
        


if __name__ == "__main__":

    # global variables
    k = 1
    i = 0
    kCounter = 0
    empty_testcase_written = False
    reach_error = False
    ntestcases = 0
    ntimesleaf = 0
    last = None

    # print version and terminate
    if len(sys.argv) == 2 and (sys.argv[1] == "-v" or sys.argv[1] == "--version"):
        print(VERSION)
        sys.exit(0)

    # set the maximum depth of the Python interpreter stack to 1,000,000 
    sys.setrecursionlimit(1000 * 1000)

    args = parse_arguments()

    # initialize the random number generator
    random.seed(args.seed)

    # handle committed arguments:
    if args.rho:
        RHO = args.rho

    source = args.file
    is_c = source[-2:] == ".c" or source[-2:] == ".i"

    if args.m32:
        BITS = 32
    elif args.m64:
        BITS = 64

    if is_c:
        binary = source[:-2]
        compile_symcc(args.library, source, binary, BITS, args.coverage)
    else:
        binary = source
        source = binary + ".c"

    if args.kExecutions:
        k = args.kExecutions

    # z3_check_sparse_models()

    stem = os.path.basename(binary)

    # define the root 
    root = Node([], "", [], [])

    # write the required metadata to the test folder
    write_metadata(source, "tests/" + stem, BITS)

    # call interrupt when the termination signal is invoked
    signal.signal(signal.SIGTERM, interrupt)


    # main loop 
    while True:
        i += 1

        # terminate when max iterations is reached
        if args.iterations and i >= args.iterations:
            print("max iterations")
            break

        if last == "error":
            break

        try:
            # root.pp()

            # 1. selection phase
            # select a suitable node -> select()
            if args.verbose:
                print("selecting")
            node = root.select(BFS)

            # is_leaf handling
            if node.is_leaf:
                ntimesleaf += 1
                if ntimesleaf == 100:
                    break
                else: 
                    continue
            
            # 2. sampling phase
            # sample the selected node -> sample()
            if args.verbose:
                print("sampling...")

            prefix = node.sample()

            if prefix is None:
                # propagate the obtained information
                node.propagate(0, 1)
                if args.verbose:
                    # sample at this path but no more choices locally (solver returned unsat), continue
                    print("e", node.path.ljust(32))
                continue
            else:
                # sample at the path, the input prefix should be shown on the right
                print("?", node.path.ljust(32), "input: " + prefix.hex())

            # manufacture requested trace constraints:
            maxlen = None

            if args.maxlen:
                maxlen = args.maxlen

            if args.adaptive or not args.maxlen:
                if i < 100:
                    maxlen = 100
                elif i < 1000:
                    maxlen = 1000
                else:
                    maxlen = 10000

            if maxlen and args.maxlen and maxlen > args.maxlen:
                maxlen = args.maxlen

            
            x = 0
            for x in range(k):
                kCounter += 1

                # 3. execution phase
                if args.verbose:
                    print("executing", binary)

                # call execute_with_input() and allocate the return values to the particular variables
                code, outs, errs, symcc_log, verifier_out = execute_with_input(
                    binary, prefix, "traces/" + stem, kCounter, args.timeout, maxlen
                )
                
                handle_execution_return_values(code, outs, errs, args.verbose)

                # 4. get trace from file
                try:
                    if args.verbose:
                        print("parse trace", symcc_log)
                    is_complete, last, trace = trace_from_file(symcc_log)
                except Exception as e:
                    node.propagate(0, 1)
                    if args.verbose:
                        print("invalid trace", e)
                    continue

                if not is_complete:
                    # node.propagate(0, 1)
                    if args.verbose:
                        print("partial trace: ", last)
                    # continue

                if not trace:
                    if not empty_testcase_written:
                        # testcov wants an empty test case
                        if args.verbose:
                            print("write empty testcase")
                        write_testcase(None, "tests/" + stem, i)
                        empty_testcase_written = True
                    if args.verbose:
                        print("deterministic")
                    continue

                bits = ["1" if bit else "0" for (_, _, bit, _) in trace]
                if args.verbose:
                    # path returned by binary execution
                    print("<", "".join(bits))

                # 5. insertion phase
                added, leaf = root.insert(trace, is_complete)
                _, _, path, _ = zip(*trace)


                # 6. propagation phase
                if added:
                    node.propagate(1, 1)
                else:
                    node.propagate(0, 1)

                if added:
                    # not executed in error-mode: write testcase
                    if not args.error or code == 1:
                        if args.verbose:
                            print("write testcase", verifier_out)
                        write_testcase(verifier_out, "tests/" + stem, kCounter)
                        ntestcases += 1
                        # new path found and integrated into the tree
                        print("+", leaf.path)

                        # executed in error-mode: only store the error testcase and terminate  
                        if last == "error" and args.error:
                            reach_error = True
                            print("reach_error() detected.")
                            break

                elif not leaf.path.startswith(node.path):
                    # path failed to preserve the prefix chosen (approximate sampler)
                    print("!", leaf.path)
                    # raise Exception("failed to preserve prefix (naive sampler is precise)")

        
        # handle exceptions or interruptions
        except KeyboardInterrupt:
            print("keyboard interrupt")
            break
        except StopIteration:
            print("signal interrupt")
            break
        except Exception as e:
            if args.verbose:
                print()
                print("current tree")
                root.pp_legend()
            raise e

    # print final output
    final_output(args, root, ntestcases, binary, reach_error, source)