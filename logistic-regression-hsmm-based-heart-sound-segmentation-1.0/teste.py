import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

eng.addpath(r"e:\Utfpr\TCC\TCC\logistic-regression-hsmm-based-heart-sound-segmentation-1.0", nargout=0)

audio_path = r"e:\Utfpr\TCC\TCC\segment_5.wav"

assigned_states = eng.segmentar_metodo(audio_path)

# assigned_states = np.array(assigned_states._data).reshape(assigned_states.size, order='F')

# print("Estados atribuídos:", assigned_states)

input("Pressione Enter para finalizar e fechar o MATLAB Engine...")