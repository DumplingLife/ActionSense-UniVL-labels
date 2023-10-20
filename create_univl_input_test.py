# create fake inputs with end = start+15, to see if wide start/end causes issue

import h5py
import numpy as np
import os
import pickle

def find_nearest(array, value):
    """Find the index of the nearest value in array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def process_hdf5_file(filepath):
    hdf_file = h5py.File(filepath, "r")

    activities_data = hdf_file["experiment-activities"]["activities"]["data"][:]
    activities_time = hdf_file["experiment-activities"]["activities"]["time_s"][:]
    frame_timestamps = hdf_file["eye-tracking-video-worldGaze"]["frame_timestamp"]["time_s"][:]

    print(filepath, len(frame_timestamps), "frames", len(frame_timestamps)/(16/10 * 29.54), "features")

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

        # remember that raw video fps is not 10, so take that into account as well
        # frames_per_feature = 16/10 * 29.54
        frames_per_feature = 10/10 * 29.54 # somehow its 10 frames per feature, not 16, weird

        start_features.append(round(start_frame/frames_per_feature))
        # end_features.append(round(stop_frame/frames_per_feature))
        end_features.append(round(start_frame/frames_per_feature) + 15)
        texts.append(caption)
        transcripts.append('none')

    return {
        'start': np.array(start_features),
        'end': np.array(end_features),
        'text': np.array(texts, dtype=object),
        'transcript': np.array(transcripts, dtype=object)
    }

output_dict = {}

# Walk through the files in the experiments directory
for dirpath, dirnames, filenames in os.walk("data/experiments"):
    for filename in filenames:
        if filename.endswith(".hdf5"):
            # Extract timestamp and subject number using a more specific split
            timestamp, tail = filename.split("_streamLog_actionNet-wearables_")
            subject = tail.split('.')[0]  # Extract the subject part, e.g., S00, from the remaining portion of the filename

            key = f"{timestamp}_{subject}_eye-tracking-video-worldGaze_frame"
            output_dict[key] = process_hdf5_file(os.path.join(dirpath, filename))

print(output_dict)

with open('data_test.pickle', 'wb') as handle:
    pickle.dump(output_dict, handle)