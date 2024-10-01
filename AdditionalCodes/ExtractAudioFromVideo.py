import moviepy.editor as mp

def extract_audio_from_video(video_path, output_audio_path):
    # Load the video file
    video_clip = mp.VideoFileClip(video_path)

    # Extract the audio
    audio_clip = video_clip.audio

    # Save the extracted audio to a file
    audio_clip.write_audiofile(output_audio_path)

    return output_audio_path

# Example usage
video_path = "sample_data/test.mp4"
output_audio_path = "sample_data/extracted_audio.wav"
audio_file = extract_audio_from_video(video_path, output_audio_path)
