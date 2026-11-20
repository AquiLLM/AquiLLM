"""Bounded POSIX pipe exchange for the isolated PDF validation process."""
import os
import selectors
import signal
import subprocess
import time


def pump(process, request, deadline, *, max_output):
    """Use one absolute deadline, bounded output, and group cleanup on failure."""
    output = bytearray()
    offset = 0
    view = memoryview(request)
    status = None
    try:
        with selectors.DefaultSelector() as selector:
            for pipe in (process.stdin, process.stdout):
                os.set_blocking(pipe.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            if request:
                selector.register(process.stdin, selectors.EVENT_WRITE)
            else:
                process.stdin.close()
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                for key, event in selector.select(remaining):
                    if event & selectors.EVENT_READ:
                        chunk = os.read(key.fd, max_output + 1 - len(output))
                        output.extend(chunk)
                        if len(output) > max_output:
                            raise OverflowError
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        try:
                            offset += os.write(key.fd, view[offset:offset + 32768])
                        except BrokenPipeError:
                            offset = len(request)
                        if offset == len(request):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            process.wait(timeout=remaining)
    except (TimeoutError, subprocess.TimeoutExpired):
        status = "timeout"
    except OverflowError:
        status = "limits"
    except Exception:
        status = "parsererror"
    finally:
        if status:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1)
        for pipe in (process.stdin, process.stdout):
            if pipe and not pipe.closed:
                pipe.close()
    return status, b"" if status else bytes(output)
