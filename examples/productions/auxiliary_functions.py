import os
import time
import socket
import platform
import subprocess
import sys
from datetime import datetime


def is_step_completed(filepath):
    """
    To skip steps based on corrupted/empty files
    Checks if the file exists and has a reasonable size (> 1KB)
    """
    return os.path.exists(filepath) and os.path.getsize(filepath) > 1024

def split_events(n_events, n_bins):
    """Splits n_events as evenly as possible across n_bins non-empty bins."""
    n_bins = max(1, min(n_bins, n_events))
    base, remainder = divmod(n_events, n_bins)
    return [base + (1 if i < remainder else 0) for i in range(n_bins)]

def merge_root_files(output_file, files):
    """Merges a list of ROOT files into output_file (via hadd) and removes the inputs."""
    if len(files) == 1:
        os.replace(files[0], output_file)
        return
    subprocess.run(["hadd", "-f", output_file] + files, check=True)
    for f in files:
        os.remove(f)

def format_time(seconds):
    """Converts seconds into a readable HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    # return f"{h}h {m}m {s}s"
    return h,m,s
    
class LorenzettiMonitor:
    def __init__(self, output_path, args):
        """
        Tracks progress of the Lorenzetti production loop 
        and writes a single consolidated status.txt.
        """

        self.args = args
        self.status_file_path = f"{output_path}/status.txt"
        self.total_iters = args.iterations

        self.global_start_time = time.time()
        self._step_start_time = self.global_start_time

        self.step_times = {
            'EVT (Generation)': [],
            'HIT (Simulation)': [],
            'ESD (Digitization)': [],
            'AOD (Reconstruction)': [],
            'NTUP (Ntuple)': []
        }

        self.current_iter = 0
        self.current_step_name = "Starting"
        self.machine_specs_dict = None
        self.finished = False

        self.machine_specs_dict = self._collect_machine_specs()

        self._write_status()
    
    ### to run on the simulation code:

    def update_live_status(self, current_iter, current_step_name):
        """Call at the start of each step."""
        self.current_iter = current_iter
        self.current_step_name = current_step_name
        self._step_start_time = time.time()
        self._write_status()
 
    def log_step_time(self):
        """Call right after a step finishes. Calculates elapsed time for a step."""
        elapsed = time.time() - self._step_start_time
        self.step_times[self.current_step_name].append(elapsed)
        print(f"Step '{self.current_step_name}' completed in {elapsed:.2f} seconds", file=sys.stderr, flush=True)
        self._write_status()
        return elapsed
 
    def mark_finished(self):
        """Call at the end of the full code."""
        self.finished = True
        self._write_status()

    ### to get the values :

    def _get_total_duration(self):
        """
        Returns formatted total duration.
        Hour, minuts, seconds returned. 
        """
        return format_time(time.time() - self.global_start_time)

    def _collect_machine_specs(self):
        """Gets the machine specs"""

        machine_specs = {}

        machine_specs['node_hostname'] = socket.gethostname()
        machine_specs['os_platform'] = platform.platform()
        machine_specs['python_version'] = sys.version.split(' ')[0]

        # cpu info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        machine_specs['cpu_model'] = line.split(':')[1].strip()
                        break
            machine_specs['total_cores'] = os.cpu_count()
        except Exception:
            machine_specs['cpu_model'] = None
        
        # memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        machine_specs['total_mem_gb'] = int(line.split()[1]) / (1024**2)
                        break
        except Exception:
            machine_specs['total_mem_gb'] = None

        return machine_specs


    def _avg_step_time(self):
        """Calculates the average time to run a specific each step."""
        avgs = {}
        for step_name, times in self.step_times.items():
            avgs[step_name] = (sum(times) / len(times)) if times else None
        return avgs
 
    def _avg_iteration_time(self):
        """Sum of the avg time of every step that has at least one data point."""
        avgs = self._avg_step_time()
        known = [v for v in avgs.values() if v is not None]
        return sum(known) if known else None

    
    def _estimated_time(self):
        avg_iter_time = self._avg_iteration_time()

        if not avg_iter_time:
            print('Not enought time samples')
            return None

        remaining_iters = self.total_iters - (self.current_iter + 1)
        estimated_seconds = remaining_iters * avg_iter_time
        return estimated_seconds
    
    ### to write each part

    def _record_live_status(self):
        lines = []
        lines.append("=========================================\n")
        lines.append("LORENZETTI LIVE STATUS DASHBOARD\n")
        lines.append("=========================================\n")
        # lines.append(f"Node Hostname: {socket.gethostname()}\n")
        lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Currently processing: Iteration {self.current_iter} (Running {self.current_step_name})\n")
        lines.append(f"Progress: {self.current_iter + 1} / {self.total_iters} Iterations\n\n")
        lines.append("")
        
        return lines

    def _record_global_metrics(self):
        lines = []
        lines.append("\n--- Global Metrics ---\n")
        from_beginning_time = time.time() - self.global_start_time
        h,m,s = format_time(from_beginning_time)
        lines.append(f"Total Time from the beginning: {h}h {m}m {s}s\n")
        lines.append("")
        return lines
    
    def _record_machine_specs(self):
        """Records the node's hardware specifications to a text file."""

        lines = []
        specs = self._collect_machine_specs()

        lines = []
        lines.append("=========================================\n")
        lines.append("MACHINE SPECIFICATIONS\n")
        lines.append("=========================================\n")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Node Hostname: {specs['node_hostname']}\n")
        lines.append(f"OS Platform: {specs['os_platform']}\n")
        lines.append(f"Python Version: {specs['python_version']}\n")
        lines.append(f"Allocated HTCondor Cores: {self.args.cores}\n")
        lines.append(f"Total Machine Cores: {specs['total_cores']}\n")
        lines.append(f"CPU Model: {specs['cpu_model']}\n")
        lines.append(f"Total Machine RAM: {specs['total_mem_gb']} GB\n")
        lines.append("")
            
        print(f"Machine specs recorded in {self.status_file_path}", file=sys.stderr)

        return lines

    def _record_arguments(self):
        lines = []
        lines.append("=========================================\n")
        lines.append("ARGUMENTS\n")
        lines.append("=========================================\n")
        for arg_name, arg_value in vars(self.args).items():
            lines.append(f"{arg_name}: {arg_value}\n")
        
        lines.append("")
        return lines
    
    def _record_raw_history(self):
        """Records the time of every individual run for each step."""
        lines = []
        lines.append("--- Raw History (All Iterations) ---")
        for step_name, times in self.step_times.items():
            if not times:
                lines.append(f"{step_name}: No data yet.")
            else:
                lines.append(f"{step_name}: {times}")
        lines.append("")
        return lines
    
    def _record_avg_time(self):
        lines = []
        lines.append("--- Average Time Per Step ---")
        for step_name, avg in self._avg_step_time().items():
            n = len(self.step_times[step_name])
            if avg is not None:
                h, m, s = format_time(avg)
                lines.append(f"{step_name}: {h}h {m}m {s}s (from {n} run(s))")
            else:
                lines.append(f"{step_name}: Waiting for data...")
        lines.append("")
        return lines
    
    def _record_time_estimate(self):
        lines = []
        lines.append("--- Time Estimate ---")
        est_time = self._estimated_time()
        if est_time is not None:
            h, m, s = format_time(est_time)
            lines.append(f"Estimated Time Remaining: ~{h}h {m}m {s}s")
            avg_iter = self._avg_iteration_time()
            est_total = avg_iter * self.total_iters
            h2, m2, s2 = format_time(est_total)
            lines.append(f"Estimated Total Time (all {self.total_iters} iterations): ~{h2}h {m2}m {s2}s")
        else:
            lines.append("Estimated Time: not enough data yet (waiting for first step to complete)")
        lines.append("")
        return lines

    ### record them all:

    def _write_status(self):
        lines = []

        lines += self._record_live_status()
        lines += self._record_global_metrics()
        lines += self._record_arguments()
        lines += self._record_machine_specs()
        lines += self._record_avg_time()
        lines += self._record_time_estimate()
        lines += self._record_raw_history()

        tmp_path = self.status_file_path + ".tmp"
        with open(tmp_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp_path, self.status_file_path)