function assigned_states = segmentar_metodo(audio_path)
    springer_options = default_Springer_HSMM_options;

    [audio_data, Fs] = audioread(audio_path);
    x = audio_data(:,1); % usa apenas um canal (mono)

    if Fs ~= 1000
        x = resample(x, 1000, Fs);
        Fs = 1000;
    end
    springer_options.audio_Fs = Fs;

    load('Springer_B_matrix.mat');
    load('Springer_pi_vector.mat');
    load('Springer_total_obs_distribution.mat');

    assigned_states = runSpringerSegmentationAlgorithm( ...
        x, Fs, B_matrix, pi_vector, total_obs_distribution, true);

    assigned_states = assigned_states + 1; % converte para 1-4

    %% --- Plotagem dos resultados ---
    t = (0:length(x)-1)/Fs;  % eixo de tempo
    
    hold on;
    plot(t, x, 'k');
    xlabel('Tempo (s)');
    ylabel('Amplitude');
    title('Segmentação do Som Cardíaco - Springer');
    
    % Cores para cada estado
    colors = [
        1 0 0;    % 1 - S1 (vermelho)
        1 0.5 0;  % 2 - Systole (laranja)
        0 0 1;    % 3 - S2 (azul)
        0 0.8 0]; % 4 - Diastole (verde)
    
    % Pinta o fundo conforme o estado atual
    yl = ylim;
    for i = 1:length(assigned_states)-1
        this_state = assigned_states(i);
        if this_state >= 1 && this_state <= 4
            patch([t(i) t(i+1) t(i+1) t(i)], ...
                  [yl(1) yl(1) yl(2) yl(2)], ...
                  colors(this_state,:), ...
                  'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
    
    % Replotar o sinal por cima[
    plot(t, x, 'k');
    legend({'Audio Data','Derived States','Sinal cardíaco','S1','Systole','S2','Diastole'});
    grid on;
    hold off;
end
