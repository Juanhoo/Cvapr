import numpy as np
import tensorflow.keras as keras
import librosa

#MODEL_PATH =
NUM_SAMPLES_TO_CONSIDER = 22050 #number of samples in teaching zone

class _keyword_spotting_Service:

    model = None
    _mapings =[
    "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    def predict(self, file_path):
        # Converting 2D into 4D arrays -> (# no_sample, # segments, #coefficients, # channels)
        MFCCs = self.preprocess(file_path)

        MFCCs = MFCCs[np.newaxis,..., np.newaxis]

        #prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc = 13, n_fft = 2048, hop_length = 512):
        # load audio file
        signal, sr = librosa.load(file_path)
        #consistency in audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc,n_fft,hop_length)

        return MFCCs.T

    # noinspection PyUnresolvedReferences
    def KeywordSpottingService():
        if _keyword_spotting_Service.instance is None:
            _keyword_spotting_Service._instance = KeywordSpottingService()
            _keyword_spotting_Service.model = keras.models.load_model(MODEL_PATH)
            return KeywordSpottingService._instance


if __name__ == "__main__":
    kss = _keyword_spotting_Service()
    kss.predict("FILE_LINK_TO_PREDICT")