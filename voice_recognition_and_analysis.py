import pyaudio
import select
import cv2
import keyboard
import argparse
import tempfile
import queue
import os
from scipy.spatial import distance as dist
import sys
from datetime import datetime
import speech_recognition as sr 
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import warnings
from pydub import AudioSegment,silence
from textblob import TextBlob
from scipy.io import wavfile
import librosa
import math
import pickle
import struct ## new
import zlib
FORMAT = pyaudio.paInt16

def voiceRecognition():
    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args()

    # AUDIO_CAPTURING
    import sounddevice as sd
    import soundfile as sf
    import numpy  # Make sure NumPy is loaded before it is used in the callback
    assert numpy  # avoid "imported but unused" message (W0611)

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='candidate_recording',
                                        suffix='.wav', dir='')
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                    channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, callback=callback):
            print('#' * 80)
            print('press q to stop the recording')
            print('#' * 80)
            # while True:
            #     file.write(q.get())
            i=0
            while True:  # making a loop
                #print(i)
                i=i+1
                file.write(q.get())
                try:  # used try so that if user pressed other than the given key error will not be shown
                    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                        print('You Pressed A Key!')
                        break  # finishing the loop
                    else:
                        pass
                except:
                    break 

# except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    #Python program to transcribe an Audio file 

    AUDIO_FILE = (args.filename) 

    # use the audio file as the audio source 

    r = sr.Recognizer() 

    with sr.AudioFile(AUDIO_FILE) as source: 
        #reads the audio file. Here we use record instead of 
        #listen 
        audio = r.record(source) 

    try: 
        text = r.recognize_google(audio)
        print("The audio file contains: " + text) 

    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio") 

    except sr.RequestError as e: 
        print("Could not request results from Google Speech Recognition service; {0}".format(e)) 


    # write audio to a WAV file
    with open("microphone-results-11223344.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # write the converted text to a TXT file
    t = open('microphone-results-11223344.txt','a')
    t.write(text)
    t.close()

    # PITCH_TRACKING
    signal = basic.SignalObj('microphone-results-11223344.wav')
    pitch = pYAAPT.yaapt(signal)
    #print(pitch.samp_values)
    print(len(pitch.samp_values))

    non_zero_pitch = []
    for i in range(len(pitch.samp_values)):
        if pitch.samp_values[i] > 0:
            non_zero_pitch.append(pitch.samp_values[i])
    print("*****************************")
    #print(non_zero_pitch)
    print(len(non_zero_pitch))

    high = []
    low = []
    for i in range(len(non_zero_pitch)):
        if non_zero_pitch[i] > 255:
            high.append(non_zero_pitch[i])
        elif non_zero_pitch[i] < 85:
            low.append(non_zero_pitch[i])


    avg_pitch = np.mean(non_zero_pitch)
    print("The average pitch value is: ", avg_pitch)

    if 85 <= avg_pitch <= 255:
        print("Appropriate Pitch Maintained", len(high), len(low))

    # GAPS_IN_AUDIO
    AudioSegment.converter = r"C:\\ffmpeg\\bin\\ffmpeg.exe"
    myaudio = AudioSegment.from_wav("microphone-results-11223344.wav")
    silent = silence.detect_silence(myaudio, min_silence_len=100, silence_thresh=-40)
    silent = [((start/1000),(stop/1000)) for start,stop in silent] #convert to sec
    print("************************")
    print(silent)
    silent = np.asarray(silent)
    print(silent)
    print(silent.shape)

    diff = []
    count = 0
    for i in range(len(silent)):
        sub = silent[i][1]-silent[i][0]
        diff.append(sub)

    for i in range(len(diff)):    
        if diff[i]  > 1.3:
            count += 1

    print("Gaps greater than 1.3 seconds: ", count, " times")

    # POLARITY_CALCULATION
    f=open("microphone-results-11223344.txt", "r")
    if f.mode == 'r':
        contents =f.read()

    blob = TextBlob(contents)
    print("The Polarity of the recorded transcript is: ")
    for sentence in blob.sentences:
        print(sentence.sentiment.polarity)

    # SPEECH_RATE
    num_words = 0 
    with open("microphone-results-11223344.txt", 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)

    print("Number of words:", num_words)
    data_, sampling_rate_ = librosa.load("microphone-results-11223344.wav", sr=44100)
    secs = np.size(data_)/sampling_rate_
    print("Audio Length: ", str(secs), " seconds")

    silent_zones = np.sum(diff)
    eff_diff = secs-silent_zones
    print("Effective non-silent time period is: ", eff_diff)

    speech_rate = math.ceil((num_words / eff_diff)*60)
    print("Speech rate is {} words per minute".format(speech_rate))

    if speech_rate < 110:
        print("Not a good speech rate: ", speech_rate)
    elif speech_rate >= 110 and speech_rate <= 165:
        print("Perfect speech rate: ", speech_rate)
    else:
        print("Very fast, either nervous or too excited: ", speech_rate)

    parser.exit(0)
	


voiceRecognition()