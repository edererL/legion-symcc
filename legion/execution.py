"""" imports """
import subprocess as sp
import threading
import z3

from legion.helper import random_bytes


def run(*args):
    """Run arguments"""
    print(*args)
    return sp.run(args, stderr=sp.STDOUT)


def compile_symcc(libs, source, binary, bits, coverage=False):
    """Compile Legion/SymCC and the required components"""

    cmd = ["clang"]
    cmd.extend(["-fpass-plugin=" + libs + "/libSymbolize.so"])

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
    """Execute the binary with samples and return obtained information"""

    def kill():
        """ Handle and kill the process"""
        
        print("killed")
        process.kill()

    sp.run(["mkdir", "-p", path])

    # inputs
    verifier_out = path + "/" + str(identifier) + ".out.txt"
    # traces
    symcc_log = path + "/" + str(identifier) + ".txt"

    env = {}
    env["VERIFIER_STDOUT"] = verifier_out
    env["SYMCC_LOG_FILE"] = symcc_log

    if timeout:
        env["SYMCC_TIMEOUT"] = str(timeout)
        timeout += 1  # let the process have 1s of graceful shutdown
    if maxlen:
        env["SYMCC_MAX_TRACE_LENGTH"] = str(maxlen)


    # create a process
    process = sp.Popen(binary, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, close_fds=True, env=env)

    # use prefix as input
    process.stdin.write(data)

    # provide random input as further necessary
    timer = threading.Timer(timeout, kill)
    try:
        timer.start()
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


def trace_from_file(trace):
    """Readout the trace from file and return is_complete, last, result and decls"""

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

                ast = "(assert " + " ".join(pending) + ")"
                constraints.append(ast)

                pending.clear()

        # readout line by line
        for line in stream.readlines():
            line = line.strip()

            if not line:
                continue
            
            last = line
            assert is_complete is None

            # handle the different components of the trace file
            if line.startswith("in  "):
                flush()

                # generate target
                k = int(line[4:])
                while nbytes < k:
                    n = "stdin" + str(nbytes)
                    x = z3.BitVec(n, 8)
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

        #ast = "\n".join(constraints)
        #constraints = constraint_from_string(ast, decls)

        for i in range(len(result)):
            (site, target, polarity) = result[i]
            result[i] = (site, target, polarity, constraints[i])

        return (is_complete, last, result, decls)


def zip_files(file, paths):
    """Zip files"""
    
    run("zip", "-r", file, *paths)
    print()