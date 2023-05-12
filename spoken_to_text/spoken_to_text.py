import speech_recognition as sr
class AudioRecorder:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def record(self, duration=5, filename="output.wav"):
        with sr.Microphone() as source:
            print("Recording...")
            audio = self.recognizer.record(source, duration=duration)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            print("Audio saved as: " + filename)

    def convert_to_text(self, wav_filename: str):
        with sr.AudioFile(wav_filename) as source:
            audio_text = self.recognizer.record(source)

        try:
            s = self.recognizer.recognize_google(audio_text)
            print("Path: ", wav_filename, " Text: " + s)
        except Exception as e:
            print("wave_path: " + wav_filename + " Exception: " + str(e))
            s = str(e)
        return s


if __name__ == '__main__':
    recorder = AudioRecorder()
    recorder.record(duration=3, filename="my_audio.wav")
    recorder.convert_to_text(wav_filename="my_audio.wav")
