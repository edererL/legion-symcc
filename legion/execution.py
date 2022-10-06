import subprocess as sp
import threading
import z3

from legion.helper import random_bytes
from legion.helper import run
from legion.helper import write_smt2_trace


def compile_symcc(libs, source, binary, bits, coverage=False):
    """Handles the compilation of the particular components"""

    cmd = ["clang"]

    cmd.extend(["-Xclang", "-load", "-Xclang", libs + "/libSymbolize.so"])

    if bits == 32:
        rpath = libs + "32"
        cmd.append("-m32")
    elif bits == 64:
        rpath = libs
        cmd.append("-m64")
    else:
        rpath = libs

    if coverage:
        cmd.append("--coverage")
    cmd.append("-fbracket-depth=1024")

    cmd.extend([source, "Verifier.cpp", "-o", binary])

    cmd.append("-lstdc++")
    cmd.append("-lm")
    cmd.append("-lSymRuntime")
    cmd.append("-L" + rpath)
    cmd.append("-Wl,-rpath," + rpath)

    run(*cmd)
    print()


def execute_with_input(binary, data, path, identifier, timeout=None, maxlen=None):
    """Executes the binary with the computed input"""

    sp.run(["mkdir", "-p", path])
    # os.remove(path)

    env = {}

    verifier_out = path + "/" + str(identifier) + ".out.txt"
    env["VERIFIER_STDOUT"] = verifier_out

    symcc_log = path + "/" + str(identifier) + ".txt"
    env["SYMCC_LOG_FILE"] = symcc_log

    if timeout:
        env["SYMCC_TIMEOUT"] = str(timeout)
        timeout += 1  # let the process have 1s of graceful shutdown
    if maxlen:
        env["SYMCC_MAX_TRACE_LENGTH"] = str(maxlen)


    # create a process
    process = sp.Popen(
        binary, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, close_fds=True, env=env
    )

    # write initial input
    process.stdin.write(data)

    def kill():
        """Kill the process"""
        print("killed")
        process.kill()

    # allows multiple tasks to run concurrently
    timer = threading.Timer(timeout, kill)
    try:
        timer.start()
        # provide random input as further necessary
        while process.poll() is None:
            try:
                process.stdin.write(random_bytes(64))
            except BrokenPipeError:
                break
    finally:
        timer.cancel()

    process.wait()

    # force reading entire output
    code = process.returncode
    outs = list(process.stdout.readlines())
    errs = list(process.stderr.readlines())

    return code, outs, errs, symcc_log, verifier_out
    

def constraint_from_string(ast, decls):
    """Return a string in SMT 2.0 format using the given sorts and decls"""
    try:
        return z3.parse_smt2_string(ast, decls=decls)
    except:
        write_smt2_trace(ast, decls, "log", "error")
        raise ValueError("Z3 parser error", ast)


def trace_from_file(trace):
    """Compute the trace from file and return is_complete, last and result"""
    with open(trace, "rt") as stream:
        nbytes = 0
        target = []
        decls = {}

        polarity = None
        site = None
        pending = []

        result = []

        constraints = []

        is_complete = None
        last = None

        def flush():
            """clear the internal buffer of the file"""
            if pending:
                # constraint = constraint_from_string(ast, decls)
                event = (site, target, polarity)
                result.append(event)

                # append the assertion to the constraint list
                ast = "(assert " + " ".join(pending) + ")"
                constraints.append(ast)

                pending.clear()

        # read and return a list of lines from the stream
        for line in stream.readlines():
            line = line.strip()

            if not line:
                continue
            
            # last stores the status (last line)
            last = line
            assert is_complete is None

            if line.startswith("in  "):
                flush()

                # number of input bytes
                k = int(line[4:])
                while nbytes < k:
                    # generate the testcase:
                    n = "stdin" + str(nbytes)
                    x = z3.BitVec(n, 8) # return a bit-vector constant named n; number of bits = 8 
                    decls[n] = x
                    target.append(x)
                    nbytes = nbytes + 1

            elif line.startswith("yes "):
                flush()
                polarity = True
                site = int(line[4:])

            elif line.startswith("no "):
                flush()
                polarity = False
                site = int(line[4:])

            elif line.startswith("exit"):
                flush()
                last = line
                is_complete = True

            elif line.startswith("error"):
                flush()
                last = line
                is_complete = True  # used by benchmark tasks

            elif line.startswith("abort"):
                flush()
                last = line
                is_complete = True  # used by benchmark tasks

            elif line.startswith("segfault"):
                flush()
                last = line
                is_complete = False

            elif line.startswith("unsupported"):
                flush()
                last = line
                is_complete = False

            elif line.startswith("timeout"):
                pending.clear()
                #flush()
                last = line
                is_complete = False

            else:
                pending.append(line)

        #flush()

        # parse all the stuff
        #ast = "\n".join(constraints)
        #constraints = constraint_from_string(ast, decls)

        for i in range(len(result)):
            (site, target, polarity) = result[i]
            result[i] = (site, target, polarity, constraints[i])

        return (is_complete, last, result, decls)