import h5py
import numpy as np

def find_nearest(array, value):
    """Find the index of the nearest value in array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


hdf_file = h5py.File("data/experiments/2022-06-14_experiment_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5", "r")

activities_data = hdf_file["experiment-activities"]["activities"]["data"][:]
activities_time = hdf_file["experiment-activities"]["activities"]["time_s"][:]
frame_timestamps = hdf_file["eye-tracking-video-worldGaze"]["frame_timestamp"]["time_s"][:]

captions = [item[0].decode('utf-8') for item in activities_data]
action = [item[1].decode('utf-8') for item in activities_data]
quality = [item[2].decode('utf-8') for item in activities_data]

# Sync timestamps to frame numbers
frame_numbers = [find_nearest(frame_timestamps, timestamp) for timestamp in activities_time]

# Pair up captions with frame numbers
result = []
for i in range(0, len(captions), 2):  # We step by 2 because of the Start/Stop pairs
    # Only process "Good"
    if quality[i] == "Good" and quality[i+1] == "Good":
        start_frame = frame_numbers[i]
        stop_frame = frame_numbers[i+1]
        
        result.append((captions[i], start_frame, stop_frame))

start_features = []
end_features = []
texts = []
transcripts = []
for item in result:
    caption, start_frame, stop_frame = item
    frames_per_feature = 16/10 * 29.54 # remember that raw video fps is not 10, so take that into account as well
    start_features.append(start_frame//frames_per_feature)
    end_features.append(stop_frame//frames_per_feature)
    texts.append(caption)
    transcripts.append('none')

# Create the final dictionary
output_dict = {
    'example_key': {
        'start': np.array(start_features),
        'end': np.array(end_features),
        'text': np.array(texts, dtype=object),
        'transcript': np.array(transcripts, dtype=object)
    }
}

print(output_dict)