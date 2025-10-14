import numpy as np
from scipy.stats import multivariate_normal
import math
from default_Springer_HSMM_options import default_Springer_HSMM_options

# Placeholders / dependências que você já tem ou irá colar:
# from your_options_module import default_Springer_HSMM_options
# from your_duration_module import get_duration_distributions

def _mnrval(coeffs, X):
    """
    Versão simples de mnrval do MATLAB.
    - coeffs pode ser:
      * 1D array (p,) -> assume regressão logística binária sem intercepto (ou intercepto incluído como coluna em X)
      * 2D array (p, K-1) -> assume forma (n_features+1, K-1) com intercepto na primeira linha (compatível com mnrval/mnrfit output)
    - X: array (T, p) ou (T, p-1) dependendo se coeffs inclui intercepto
    Retorna: probs (T, K) com K classes (K=2 para binário)
    """
    X = np.asarray(X)
    coeffs = np.asarray(coeffs)

    # Caso binário simples: coeffs 1D
    if coeffs.ndim == 1:
        # se X inclui coluna de 1s para intercepto, ok; senão treat X @ coeffs
        logits = X.dot(coeffs)
        p = 1.0 / (1.0 + np.exp(-logits))
        probs = np.vstack([1 - p, p]).T
        return probs

    # Caso multiclasses: coeffs shape (p, K-1)
    if coeffs.ndim == 2:
        # linear predictors for K-1 classes
        # se X shape (T,p) deve ser compatível
        linpred = X.dot(coeffs)  # (T, K-1)
        # adiciona coluna de zeros para classe de referência
        linpred_full = np.concatenate([np.zeros((linpred.shape[0], 1)), linpred], axis=1)
        # softmax
        e = np.exp(linpred_full - np.max(linpred_full, axis=1, keepdims=True))
        probs = e / np.sum(e, axis=1, keepdims=True)
        return probs

    # fallback
    T = X.shape[0]
    return np.ones((T, 2)) * 0.5


def _mvnpdf(x, mean, cov):
    """
    PDF multivariada. x shape (d,) or (n,d); mean (d,) ; cov (d,d) or scalar.
    Retorna vector (n,)
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    # Se x é 1D -> torna (1,d)
    single = False
    if x.ndim == 1:
        x = x[np.newaxis, :]
        single = True

    # Cov pode vir como escalar (variance)
    try:
        rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        vals = rv.pdf(x)
    except Exception:
        # fallback: assumindo cov diagonal
        if np.isscalar(cov):
            var = float(cov)
            d = mean.size
            denom = np.sqrt((2 * np.pi * var) ** d)
            dif = x - mean
            ex = np.exp(-0.5 * np.sum(dif ** 2, axis=1) / var)
            vals = ex / denom
        else:
            # última tentativa: usar diagonal da cov
            diag = np.diag(cov)
            var = diag
            denom = np.sqrt(np.prod(2 * np.pi * var))
            dif = x - mean
            ex = np.exp(-0.5 * np.sum((dif ** 2) / (var + 1e-12), axis=1))
            vals = ex / (denom + 1e-12)
    return vals if not single else vals[0]


def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs, figures=False):
    """
    Tradução python de viterbiDecodePCG_Springer.m
    observation_sequence: ndarray (T, n_features)
    pi_vector: array-like (N,)
    b_matrix: list/array of coefficient matrices for mnrval (length N)
    total_obs_distribution: tuple/list (mean, cov) OR structure similar ao MATLAB cell
    heartrate: scalar (bpm)
    systolic_time: scalar (s)
    Fs: sampling freq used for durations (features sampling freq expected)
    figures: bool (plot placeholders)
    Returns: delta, psi, qt
    """
    # ---------- placeholders (deixe por onde você preferir importar) ----------
    try:
        springer_options = default_Springer_HSMM_options()
    except Exception:
        springer_options = {'use_mex': False}

    # get_duration_distributions must be provided (project-specific)
    try:
        d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole = get_duration_distributions(heartrate, systolic_time)
    except Exception:
        # placeholder simple guess -> gaussians around heuristic durations (in samples)
        # WARNING: você deve substituir por sua função real
        # estimativas baseadas em heartrate e systolic_time
        Tdummy = int(round((60.0/heartrate) * Fs))
        mu = np.array([0.15*Fs, systolic_time*Fs, 0.15*Fs, 0.55*(60.0/heartrate)*Fs])  # guesses
        sigma = np.maximum(1.0, 0.1 * mu)
        # d_distributions as array-like of (mean, var)
        d_distributions = np.empty((4,2), dtype=object)
        for j in range(4):
            d_distributions[j,0] = mu[j]
            d_distributions[j,1] = sigma[j]**2
        # rough mins/maxs:
        max_S1 = int(mu[0] + 3*sigma[0]); min_S1 = int(max(1, mu[0] - 3*sigma[0]))
        max_S2 = int(mu[2] + 3*sigma[2]); min_S2 = int(max(1, mu[2] - 3*sigma[2]))
        max_systole = int(mu[1] + 3*sigma[1]); min_systole = int(max(1, mu[1] - 3*sigma[1]))
        max_diastole = int(mu[3] + 3*sigma[3]); min_diastole = int(max(1, mu[3] - 3*sigma[3]))

    # ---------- init ----------
    observation_sequence = np.asarray(observation_sequence)
    T = observation_sequence.shape[0]
    N = 4

    max_duration_D = int(round((1.0 * (60.0 / heartrate)) * Fs))

    NEG_INF = -1e300
    delta = np.ones((T + max_duration_D - 1, N)) * NEG_INF
    psi = np.zeros((T + max_duration_D - 1, N), dtype=int)
    psi_duration = np.zeros((T + max_duration_D - 1, N), dtype=int)

    # ---------- observation probabilities ----------
    observation_probs = np.zeros((T, N))
    # b_matrix: assume iterable of length N with coefficient arrays for mnrval
    for n in range(N):
        coeffs_n = b_matrix[n]
        # pihat shape (T, K)
        pihat = _mnrval(coeffs_n, observation_sequence)
        # total_obs_distribution: expect (mean, cov) or similar
        for t in range(T):
            # Po_correction = mvnpdf(observation_sequence(t,:), total_obs_distribution.mean, total_obs_distribution.cov)
            try:
                mean_tot = total_obs_distribution[0]
                cov_tot = total_obs_distribution[1]
            except Exception:
                mean_tot = np.zeros(observation_sequence.shape[1])
                cov_tot = np.eye(observation_sequence.shape[1])
            Po_correction = _mvnpdf(observation_sequence[t, :], mean_tot, cov_tot)
            # note: MATLAB picks pihat(t,2) for class 1 prob (indexing)
            # Here if pihat has >=2 cols, use column 1 (the second)
            if pihat.shape[1] >= 2:
                p_state_given_obs = pihat[t, 1]
            else:
                # binary case: pihat[:,1] is p for class 1
                p_state_given_obs = pihat[t, -1]
            # Avoid division by zero for pi_vector element
            denom = pi_vector[n] if pi_vector[n] != 0 else 1e-12
            observation_probs[t, n] = (p_state_given_obs * Po_correction) / denom

    # ---------- duration probabilities ----------
    # duration_probs shape (N, max_duration_D) but MATLAB preallocates larger
    max_len = max(3 * Fs, max_duration_D)
    duration_probs = np.zeros((N, max_len))
    duration_sum = np.zeros(N)

    for j in range(N):
        for d in range(1, max_duration_D + 1):
            # fetch mean/var from d_distributions: expected format similar to MATLAB cell
            try:
                mu_d = float(d_distributions[j, 0])
                var_d = float(d_distributions[j, 1])
            except Exception:
                # fallback: small variance
                mu_d = float(d_distributions[j, 0])
                var_d = float(d_distributions[j, 1]) if d_distributions[j, 1] != 0 else 1.0

            # mvnpdf of scalar d ~ N(mu_d, var_d)
            # use univariate normal pdf
            denom = math.sqrt(2 * math.pi * var_d) + 1e-12
            exponent = -0.5 * ((d - mu_d) ** 2) / (var_d + 1e-12)
            duration_probs[j, d - 1] = (math.exp(exponent) / denom)

            # Apply hard min/max constraints based on MATLAB code:
            if j == 0:
                if (d < min_S1) or (d > max_S1):
                    duration_probs[j, d - 1] = np.finfo(float).tiny
            elif j == 2:
                if (d < min_S2) or (d > max_S2):
                    duration_probs[j, d - 1] = np.finfo(float).tiny
            elif j == 1:
                if (d < min_systole) or (d > max_systole):
                    duration_probs[j, d - 1] = np.finfo(float).tiny
            elif j == 3:
                if (d < min_diastole) or (d > max_diastole):
                    duration_probs[j, d - 1] = np.finfo(float).tiny

        duration_sum[j] = np.sum(duration_probs[j, :max_duration_D])

    # trim to 3*Fs if longer (as in MATLAB)
    if duration_probs.shape[1] > 3 * Fs:
        duration_probs = duration_probs[:, :int(3 * Fs)]

    # figures (optional) - placeholder: user can implement plotting outside
    if figures:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            for j in range(N):
                plt.plot(duration_probs[j, :int(max_duration_D)] / (duration_sum[j] + 1e-12), label=f'state {j+1}')
            plt.legend(['S1', 'Systole', 'S2', 'Diastole'])
            plt.title('Duration probabilities')
            plt.show()
        except Exception:
            pass

    # ---------- Viterbi recursion (extended duration HMM) ----------
    # Initialization
    # delta(1,:) = log(pi_vector) + log(observation_probs(1,:))
    # careful with zeros -> add tiny
    tiny = 1e-300
    delta[0, :] = np.log(np.maximum(pi_vector, tiny)) + np.log(np.maximum(observation_probs[0, :], tiny))
    psi[0, :] = -1

    # transition matrix a_matrix (forced cycle 1->2->3->4->1)
    a_matrix = np.array([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]], dtype=float)

    # Recursion
    for t in range(1, T + max_duration_D - 1):  # MATLAB range 2 : T+max_duration_D-1
        for j in range(N):
            # we will examine all durations d=1..max_duration_D
            best_val = NEG_INF
            best_prev = 0
            best_d = 1
            # compute end index (end_t) = min(t, T-1) in MATLAB; careful with indices
            for d in range(1, max_duration_D + 1):
                start_t = t - d + 1  # MATLAB start_t = t - d + 1 when using 1-based; but their code used start_t=t-d then clamp+1; adjust
                if start_t < 0:
                    start_t = 0
                if start_t > T - 2:
                    start_t = T - 2

                end_t = t
                if t >= T:
                    end_t = T - 1

                # find max over previous states of delta(start_t, :) + log(a(:,j))'
                # note: a_matrix[:,j] is column of transitions to j
                # delta[start_t, :] shape (N,)
                trans = a_matrix[:, j]
                # add log, but trans can be zero -> log(0) = -inf which is fine
                with np.errstate(divide='ignore'):
                    log_trans = np.log(trans + 1e-300)
                prev_vals = delta[start_t, :] + log_trans
                max_delta = np.max(prev_vals)
                max_index = np.argmax(prev_vals)

                # emission probs: product of observation_probs[start_t:end_t+1, j]
                # in MATLAB they do prod(observation_probs(start_t:end_t,j))
                # convert to avoid underflow: sum of logs
                obs_slice = observation_probs[start_t:end_t+1, j]
                if obs_slice.size == 0:
                    probs = tiny
                else:
                    probs = np.prod(obs_slice)
                if probs == 0:
                    probs = np.finfo(float).tiny
                emission_log = np.log(probs)

                # duration probability
                dur = d
                # duration_probs[j, d-1] ... duration_sum[j]
                dur_prob = duration_probs[j, d - 1] if (d - 1) < duration_probs.shape[1] else np.finfo(float).tiny
                denom_d = duration_sum[j] if duration_sum[j] != 0 else 1e-12
                with np.errstate(divide='ignore'):
                    log_dur = np.log(dur_prob / denom_d + 1e-300)

                delta_temp = max_delta + emission_log + log_dur

                if delta_temp > delta[t, j]:
                    delta[t, j] = delta_temp
                    psi[t, j] = max_index
                    psi_duration[t, j] = d

    # ---------- Termination & backtracking ----------
    # temp_delta = delta[T+1:end, :]
    temp_delta = delta[T:, :]
    # find max value and position
    flat_idx = np.argmax(temp_delta)
    pos_rel, state = np.unravel_index(flat_idx, temp_delta.shape)
    pos = pos_rel + T  # convert to MATLAB-like index

    # convert pos to 0-based index into delta
    # Note: pos is index in delta array (0-based) because we used 0-based everywhere
    # state is 0-based index
    offset = pos
    preceding_state = int(psi[offset, state])

    onset = offset - int(psi_duration[offset, state]) + 1
    if onset < 0:
        onset = 0

    qt = np.zeros(T, dtype=int)

    # fill qt[onset:offset] with state+1 (to keep 1..4 like MATLAB)
    # clamp offset to within T-1
    off_clamped = min(offset, T - 1)
    qt[onset:off_clamped + 1] = state + 1

    state = preceding_state
    count = 0

    while onset > 1:
        offset = onset - 1
        preceding_state = int(psi[offset, state])
        onset = offset - int(psi_duration[offset, state]) + 1
        if onset < 0:
            onset = 0
        off_clamped = min(offset, T - 1)
        qt[onset:off_clamped + 1] = state + 1
        state = preceding_state
        count += 1
        if count > 1000:
            break

    # trim qt to length T (already)
    return delta, psi, qt
