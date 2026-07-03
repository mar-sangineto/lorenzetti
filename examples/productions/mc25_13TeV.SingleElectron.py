import os
import sys
import subprocess
import argparse
import time # can be present just on auxiliary functions
import numpy as np
from datetime import datetime

from auxiliary_functions import *
from energy_samplers import get_energies, SUPPORTED_DISTRIBUTIONS

start_time = time.time() # can be replaced by the existing one on the auxiliary functions

now = datetime.now()
run_number = now.strftime("%Y%m%d")

parser = argparse.ArgumentParser(description="Script for job execution.")

parser.add_argument('--run-name', type=str, default="", help="Run name") # to not overwrite or to continue training
parser.add_argument('--out-path', type=str, default='.', help="Path to save the results")
parser.add_argument('--cores', type=int, default=1, help="Number of allocated cores")
parser.add_argument('--iterations', type=int, default=15, help="Total number of iterations (default: 15)")
parser.add_argument('--events-number', type=int, default=2000, help="Number of events per iteration (default: 2000)")
parser.add_argument('--last-step', type=int, default=2, choices=range(5),
                    help="Last pipeline step to run. 0-EVT 1-HIT 2-ESD 3-AOD 4-NTUP (default: 2-ESD)")

# particle gun kinematics:
parser.add_argument('--eta-min', type=float, default=-2.5, help="Minimum eta (default: -2.5)")
parser.add_argument('--eta-max', type=float, default=2.5, help="Maximum eta (default: 2.5)")
parser.add_argument('--phi-min', type=float, default=-3.1415, help="Minimum phi (default: -3.1415)")
parser.add_argument('--phi-max', type=float, default=3.1415, help="Maximum phi (default: 3.1415)")
parser.add_argument('--energy-min', type=float, default=2, help="Minimum energy in GeV (default: 2)")
parser.add_argument('--energy-max', type=float, default=7000, help="Maximum energy in GeV (default: 7000)")
parser.add_argument('--energy-dist', type=str, default="linear", choices=SUPPORTED_DISTRIBUTIONS,
                    help=f"Shape of the energy spectrum to sample from, i.e. how non-homogeneously "
                         f"energies are distributed (default: log). Choices: {SUPPORTED_DISTRIBUTIONS}")
parser.add_argument('--energy-bins', type=int, default=50,
                    help="Number of distinct energy values drawn from --energy-dist per iteration; "
                         "events are split evenly across them to build the energy spectrum (default: 50)")

# cluster arguments:
parser.add_argument('--proc-id', type=int, default=0, help="Current job ID") # in case of parallelizing the iteration in the cluster
parser.add_argument('--cluster-id', type=int, default=0, help="Cluster submission ID (prevents overwriting runs)") # for the cluster working in parallel queue instead of the iteration

args = parser.parse_args()

allocated_cores = args.cores
iterations_number = args.iterations
last_step_idx = args.last_step

events_per_chunk = args.events_number
run_name = args.run_name if args.run_name else now.strftime("%Y%m%d_%H%M")

if args.cluster_id:
    run_name = f"{run_name}_c{args.cluster_id}_{args.proc_id}"

output_path=f"{args.out_path}/{run_name}"
os.makedirs(output_path, exist_ok=True)

monitor = LorenzettiMonitor(output_path,args)

print("Allocated cores = ", allocated_cores, "Number of iterations = ", iterations_number,\
        "events per iteration ", events_per_chunk,\
        "saved on ",output_path, file=sys.stderr, flush=True)

# Create folder structure for each stage
for i in range(5):
    os.makedirs(f"{output_path}/step_{i}", exist_ok=True)

print(f"=== Starting production of {iterations_number} files with {events_per_chunk} events ===")

for iteration in range(iterations_number):
    if iterations_number==1:
        iteration = args.proc_id
    print(f"\n==========================================")
    print(f" PROCESSING ITERATION {iteration + 1} / {iterations_number}")
    print(f"==========================================")
    
    # Calculate a unique seed for each iteration or job based on ID
    seed = args.cluster_id * 1_000_000 + args.proc_id * 10_000 + (1 + iteration)
    monitor.log_master_seed(iteration, seed)
    
    # Definition of file names for this specific iteration
    evt_file = f"{output_path}/step_0/Electron.EVT.{iteration}.root"
    hit_file = f"{output_path}/step_1/Electron.HIT.{iteration}.root"
    esd_file = f"{output_path}/step_2/Electron.ESD.{iteration}.root"
    aod_file = f"{output_path}/step_3/Electron.AOD.{iteration}.root"
    ntup_file = f"{output_path}/step_4/Electron.{iteration}.root"

    # --- STEP 0: GENERATION (EVT) ---
    if last_step_idx >= 0:
        monitor.update_live_status(iteration, "EVT (Generation)")
        if is_step_completed(evt_file):
            print("Generation already completed")
        else:
            print(f"-> [1/5] Running Generation (gen_single.py) for iteration {iteration}...")

            if args.energy_dist == "linear":
                cmd_evt = [
                    "gen_single.py", "-p", "Electron", "-o", evt_file,
                    "--nov", str(events_per_chunk), "--events-per-job", str(events_per_chunk),
                    "--do-eta-ranged", str(args.eta_min==args.eta_max), "--do-phi-ranged", str(args.phi_min==args.phi_max),
                    "--eta-min", str(args.eta_min), "--eta-max", str(args.eta_max),
                    "--phi-min", str(args.phi_min), "--phi-max", str(args.phi_max),
                    "--energy-min", str(args.energy_min), "--energy-max", str(args.energy_max),
                    "--energy", "0",
                    "-s", str(seed), "--run-number", str(run_number)
                ]
                subprocess.run(cmd_evt, check=True)
            else:
                # gen_single.py only supports a single fixed energy (or a flat range) per
                # call, so a non-homogeneous spectrum is built here by drawing --energy-bins
                # fixed energies from --energy-dist and splitting the events across them.
                np.random.seed(seed)
                n_bins = max(1, min(args.energy_bins, events_per_chunk))
                bin_energies = get_energies(args.energy_dist, args.energy_min, args.energy_max, n_bins)
                bin_events = split_events(events_per_chunk, n_bins)

                bin_files = []
                for bin_idx, (bin_energy, bin_nov) in enumerate(zip(bin_energies, bin_events)):
                    if bin_nov == 0:
                        continue
                    bin_file = f"{output_path}/step_1/Electron.EVT.{iteration}.bin{bin_idx}.root"
                    bin_seed = seed * 1000 + bin_idx
                    monitor.log_bin_seed(iteration, bin_idx, bin_seed, bin_energy)
                    cmd_evt = [
                        "gen_single.py", "-p", "Electron", "-o", bin_file,
                        "--nov", str(bin_nov), "--events-per-job", str(bin_nov),
                        "--do-eta-ranged", str(args.eta_min==args.eta_max), "--do-phi-ranged", str(args.phi_min==args.phi_max),
                        "--eta-min", str(args.eta_min), "--eta-max", str(args.eta_max),
                        "--phi-min", str(args.phi_min), "--phi-max", str(args.phi_max),
                        "--energy", str(bin_energy),
                        "-s", str(bin_seed), "--run-number", str(run_number)
                    ]
                    subprocess.run(cmd_evt, check=True)
                    bin_files.append(bin_file)

                merge_root_files(evt_file, bin_files)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)


    # --- STEP 1: DETECTOR SIMULATION (HIT) ---
    if last_step_idx >= 1:
        monitor.update_live_status(iteration, "HIT (Simulation)")
        if is_step_completed(hit_file):
            print("HIT already done, skipping this part")
        else:
            print(f"-> [2/5] Running Simulation (simu_trf.py) for iteration {iteration}...")
            cmd_hit = ["simu_trf.py", "-i", evt_file, "-o", hit_file, "-nt", str(allocated_cores)]
            subprocess.run(cmd_hit, check=True)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)

    # --- STEP 2: DIGITALIZAÇÃO (ESD) ---
    if last_step_idx >= 2:
        monitor.update_live_status(iteration, "ESD (Digitization)")
        if is_step_completed(esd_file):
            print("ESD already completed")
        else:
            print(f"-> [3/5] Running Digitization (digit_trf.py) for iteration {iteration}...")
            cmd_esd = ["digit_trf.py", "-i", hit_file, "-o", esd_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
            subprocess.run(cmd_esd, check=True)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)

    # --- STEP 3: RECONSTRUCTION (AOD) ---
    if last_step_idx >= 3:
        monitor.update_live_status(iteration, "AOD (Reconstruction)")
        if is_step_completed(aod_file):
            print("AOD already completed")
        else:
            print(f"-> [4/5] Running Reconstruction (reco_trf.py) for iteration {iteration}...")
            cmd_aod = ["reco_trf.py", "-i", esd_file, "-o", aod_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
            subprocess.run(cmd_aod, check=True)
            monitor.log_step_time()

    # --- STEP 4: FINAL NTUPLE ---
    if last_step_idx >= 4:
        monitor.update_live_status(iteration, "NTUP (Ntuple)")
        if is_step_completed(ntup_file):
            print("NTUP already completed")
        else:
            print(f"-> [5/5] Generating Final Ntuple (ntuple_trf.py) for iteration {iteration}...")
            cmd_ntup = ["ntuple_trf.py", "-i", aod_file, "-o", ntup_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
            subprocess.run(cmd_ntup, check=True)
        monitor.log_step_time()

end_time = time.time()
total_duration = end_time - start_time
total_events = iterations_number * events_per_chunk

# Convert duration to hours, minutes, and seconds for better readability
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

print(f"\n=======================================================")
print(f" SUCCESS! ALL {iterations_number} JOBS HAVE BEEN COMPLETED!")
print(f"=======================================================")
print(f" Total Jobs Processed: {iterations_number}")
print(f" Total Events Generated: {total_events}")
print(f" Total Simulation Time: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)")
print(f"=======================================================")

