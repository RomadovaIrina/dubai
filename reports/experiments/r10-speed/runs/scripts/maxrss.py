import resource, subprocess, sys, time
t = time.time(); rc = subprocess.call(sys.argv[1:])
print(f"Maximum resident GB {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6:.1f} Elapsed {time.time() - t:.0f} s rc {rc}", flush=True)
