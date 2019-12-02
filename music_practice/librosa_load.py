import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_name = "../note_detection/violin2.wav"

'''
def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y
'''
def load_file(file_name):
    y, sr = librosa.load(file_name)
    return y, sr

def get_duration():
    duration = int(librosa.get_duration(y=y, sr=sr))
    return duration

# load file from librosa
y, sr = load_file(file_name)

duration = get_duration()
print ('duration :', str(duration))

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# can save with csv
#librosa.output.times_csv('beat_times.csv', beat_times)
print (beat_times)


plt.plot(y)
plt.savefig('test.png')
plt.show()
