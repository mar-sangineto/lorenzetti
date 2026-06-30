import os
import sys
import subprocess
import argparse
import time # can be present just on auxiliary functions
from datetime import datetime

from auxiliary_functions import *

start_time = time.time() # can be replaced by the existing one on the auxiliary functions

now = datetime.now()
run_number = now.strftime("%Y%m%d_%H%M")

parser = argparse.ArgumentParser(description="Script for job execution.")

parser.add_argument('--run-name', type=str, default="", help="Run name") # to not overwrite or to continue training
parser.add_argument('--out-path', type=str, default='./', help="Path to save the results")
parser.add_argument('--cores', type=int, default=1, help="Number of allocated cores")
parser.add_argument('--iterations', type=int, default=15, help="Total number of iterations (default: 15)")
parser.add_argument('--events-number', type=int, default=2000, help="Number of events per iteration (default: 2000)")

# cluster arguments:
parser.add_argument('--proc-id', type=int, default=1, help="Current job ID") # in case of parallelizing the iteration in the cluster
parser.add_argument('--cluster-id', type=str, default="", help="Cluster submission ID (prevents overwriting runs)") # for the cluster working in parallel queue instead of the iteration

args = parser.parse_args()

allocated_cores = args.cores
iterations_number = args.iterations

events_per_chunk = args.events_number
run_name = args.run_name if args.run_name else run_number

if args.cluster_id:
    run_name = f"{run_name}_c{args.cluster_id}_{args.proc_id}"

output_path=f"{args.out_path}/{run_name}"
os.makedirs(output_path, exist_ok=True)

monitor = LorenzettiMonitor(output_path,args)

# print("Allocated cores = ", allocated_cores, "Number of iterations = ", iterations_number,\
#         "events per iteration ", events_per_chunk,\
#         "saved on ",output_path, file=sys.stderr, flush=True)

# Create folder structure for each stage
stages = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5']
for stage in stages:
    os.makedirs(f"{output_path}/{stage}", exist_ok=True)

# def is_step_completed(filepath):
#     """
#     To skip steps based on corrupted/empty files
#     Checks if the file exists and has a reasonable size (> 1KB)
#     """
#     return os.path.exists(filepath) and os.path.getsize(filepath) > 1024

print(f"=== Starting production of {iterations_number} files with {events_per_chunk} events ===")

for iteration in range(iterations_number):
    if iterations_number==1:
        iteration = args.proc_id
    print(f"\n==========================================")
    print(f" PROCESSING ITERATION {iteration} / {iterations_number - 1}")
    print(f"==========================================")
    
    # Calculate a unique seed for each iteration or job based on ID
    seed = 16 * (1 + iteration)
    
    # Definition of file names for this specific iteration
    evt_file = f"{output_path}/step_1/Electron.EVT.{iteration}.root"
    hit_file = f"{output_path}/step_2/Electron.HIT.{iteration}.root"
    esd_file = f"{output_path}/step_3/Electron.ESD.{iteration}.root"
    aod_file = f"{output_path}/step_4/Electron.AOD.{iteration}.root"
    ntup_file = f"{output_path}/step_5/Electron.{iteration}.root"

    # --- STEP 1: GENERATION (EVT) ---
    monitor.update_live_status(iteration, "EVT (Generation)")
    if is_step_completed(evt_file):
        print("Generation already completed")
    else:
        print(f"-> [1/5] Running Generation (gen_single.py) for iteration {iteration}...")
        cmd_evt = [
                "gen_single.py", "-p", "Electron", "-o", evt_file,
            "--nov", str(events_per_chunk), "--events-per-job", str(events_per_chunk),
            "--do-eta-ranged", "a", "--do-phi-ranged", "a",
            "--eta-min", "-2.5", "--eta-max", "2.5",
            "--phi-min", "-3.1415", "--phi-max", "3.1415",
            "--energy-min", "2", "--energy-max", "7000", "--energy", "0",
            "-s", str(seed), "--run-number", str(run_number)
        ]
        subprocess.run(cmd_evt, check=True)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)


    # --- STEP 2: DETECTOR SIMULATION (HIT) ---
    monitor.update_live_status(iteration, "HIT (Simulation)")
    if is_step_completed(hit_file):
        print("HIT already done, skipping this part")
    else:
        print(f"-> [2/5] Running Simulation (simu_trf.py) for iteration {iteration}...")
        cmd_hit = ["simu_trf.py", "-i", evt_file, "-o", hit_file, "-nt", str(allocated_cores)]
        subprocess.run(cmd_hit, check=True)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)

    # --- STEP 3: DIGITALIZAÇÃO (ESD) ---
    monitor.update_live_status(iteration, "ESD (Digitization)")
    if is_step_completed(esd_file):
        print("ESD already completed")
    else:
        print(f"-> [3/5] Running Digitization (digit_trf.py) for iteration {iteration}...")
        cmd_esd = ["digit_trf.py", "-i", hit_file, "-o", esd_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
        subprocess.run(cmd_esd, check=True)
        monitor.log_step_time()
        print(f"it tooked {time.time()-start_time}", file=sys.stderr, flush=True)

    # --- STEP 4: RECONSTRUCTION (AOD) ---
    # monitor.update_live_status(iteration, "AOD (Reconstruction)")
    # print(f"-> [4/5] Running Reconstruction (reco_trf.py) for iteration {iteration}...")
    # cmd_aod = ["reco_trf.py", "-i", esd_file, "-o", aod_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
    # subprocess.run(cmd_aod, check=True)
    # monitor.log_step_time()

    # # --- STEP 5: FINAL NTUPLE ---
    # monitor.update_live_status(iteration, "NTUP (Ntuple)")
    # print(f"-> [5/5] Generating Final Ntuple (ntuple_trf.py) for iteration {iteration}...")
    # cmd_ntup = ["ntuple_trf.py", "-i", aod_file, "-o", ntup_file, "-nt", str(allocated_cores), "--events-per-job", str(events_per_chunk), "-m"]
    # subprocess.run(cmd_ntup, check=True)
    # monitor.log_step_time()

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

