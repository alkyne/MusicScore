import librosa
import librosa.display
import timeit
import numpy as np
import matplotlib.pyplot as plt

def load_file(audio_name):

    #audio_name = 'test.mp3'
    #audio_path = './' + audio_name

    sample_rate = 44100     # usually use 16000, 22050, 44100, 48000, etc.
    is_stereo = True        # if wave file is stereo

    start = timeit.default_timer()
    wave, sr = librosa.load(audio_name)#, sample_rate, mono=not is_stereo)
    stop = timeit.default_timer()

    print('Successfully loaded wave: {}'.format(audio_name))
    print('Elapsed time: {:.4g}s'.format(stop - start))
    print('shape: {}'.format(wave.shape))       # loaded wave in librosa is Numpy array

    return wave, sr

if __name__ == '__main__':
    wave, sr = load_file('./test.mp3')
    D = np.abs(librosa.stft(wave)) # fft
    print(D)

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title('FFT')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('test.png')
    plt.show()
    