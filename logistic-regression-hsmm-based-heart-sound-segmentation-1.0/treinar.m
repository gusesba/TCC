close all;
clear all;

springer_options = default_Springer_HSMM_options;

load('example_data.mat');

training_indices = 1:length(example_data.example_audio_data);
train_recordings = example_data.example_audio_data(training_indices);
train_annotations = example_data.example_annotations(training_indices,:);

%% Train the HMM:
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

%% Salva os arquivos .mat para uso posterior
save('Springer_B_matrix.mat', 'B_matrix');
save('Springer_pi_vector.mat', 'pi_vector');
save('Springer_total_obs_distribution.mat', 'total_obs_distribution');

disp('Arquivos do modelo Springer gerados com sucesso!');
