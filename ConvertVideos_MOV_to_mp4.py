#The provided code is a Python script for converting a list of video files into .mp4 format using MoviePy and parallelizing the task with the multiprocessing module.
# This script is suitable for bulk video format conversion, particularly when processing a large number of files, as it significantly reduces processing time by running conversions in parallel.


import moviepy.editor as moviepy
import os
from multiprocessing import Pool, cpu_count

# List of input video file paths
video_files = ['Scenario2_Ipad1_01.MOV', 'Scenario2_Ipad1_05.MOV', 'Scenario2_Ipad2_01.MOV', 'Script1_Ipad2_06.MOV']

# Output directory for the converted videos
output_dir = "CPU_ConvertedVideos/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the video conversion function to be run in parallel
def convert_video(video_file):
    try:
        # Load the video clip
        clip = moviepy.VideoFileClip(video_file)

        # Extract the base filename without the extension
        base_name = os.path.splitext(os.path.basename(video_file))[0]

        # Define the output file path with the new extension
        output_file = os.path.join(output_dir, f"{base_name}.mp4")

        # Write the video file in mp4 format
        clip.write_videofile(output_file)

        print(f"Converted {video_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {video_file}: {e}")

# Main execution to parallelize the conversion
if __name__ == "__main__":
    # Determine the number of CPU cores available
    num_cores = cpu_count()-1

    # Create a Pool of workers based on the number of CPU cores
    with Pool(num_cores) as pool:
        # Use the pool to map the conversion function to video files
        pool.map(convert_video, video_files)

