import moviepy.editor as moviepy

clip = moviepy.VideoFileClip("/Users/pallavigupta/Documents/Multimodal_LMS/scripts/Videos/01.MOV")

clip.write_videofile("test.mp4")
