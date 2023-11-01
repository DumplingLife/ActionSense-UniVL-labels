import h5py
import numpy as np

# Paths
hdf5_path = "data/experiments/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"
output_folder = "emg_chunks/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00/"

# Load HDF5 file
with h5py.File(hdf5_path, "r") as f:
    # Load EMG data and timestamps
    emg_data = f["myo-right/emg/data"][:]
    emg_time = f["myo-right/emg/time_s"][:]
    
    # Load video frame timestamps
    frame_time = f["eye-tracking-video-worldGaze/frame_timestamp/time_s"][:]

    chunk_times = 16.66667
    
    # Initialize variables
    start_time = frame_time[0]
    end_time = frame_time[-1]
    print(start_time)
    print(end_time)
    current_chunk_start = start_time
    current_chunk_end = current_chunk_start + chunk_times
    
    chunk_index = 0
    emg_chunk = []
    
    # Loop through EMG data
    for time, data in zip(emg_time, emg_data):
        if time >= current_chunk_start and time < current_chunk_end:
            emg_chunk.append(data)
        elif time >= current_chunk_end:
            # Save current chunk
            print(time)
            np.save(f"{output_folder}/emg_{str(chunk_index).zfill(3)}.npy", np.array(emg_chunk))
            
            # Update variables for next chunk
            chunk_index += 1
            current_chunk_start = current_chunk_end
            current_chunk_end = current_chunk_start + chunk_times
            
            # Initialize new chunk
            emg_chunk = [data]
    
    # Save the last chunk
    np.save(f"{output_folder}/emg_{str(chunk_index).zfill(3)}.npy", np.array(emg_chunk))
