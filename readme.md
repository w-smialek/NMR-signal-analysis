## Effects of frequency non-stationarity on spectra

The simplest examples concern 1D spectrum with time varying frequency.
If the time over witch the change of frequency takes place is short in some sense, we can focus only on the linear change of frequency with time:
$$S_k(t) = A_k \cdot e^{2\pi i \ t \cdot f_k(t) - t/\tau_k}$$
$$f_k(t) = u+w \cdot t$$
The spectrum of $S_k(t)$ will no longer be a Lorentzian, but instead will contain oscillations in the range of frequencies present in the time signal.
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main//1d_nonstat_example.png)
Image shows the real part of the Discrete Fourier Transform of a signal $S_k(t) = e^{2\pi i\cdot(50+2.5\frac{t}{512})t - t/0.5}$ plus a complex white noise with variance 0.1, with 512 sampling points and sampling rate $\Delta t = \frac {1}{512} \text{s}$

The corresponding behaviour is observed in 2D spectra. For the real-life spectroscopy experiment, we assume the following setting:
- There is a series of 1D direct time $t_1$ signals registered, with varying indirect time value $t_2$. 
- Direct time is always fully sampled, with $t_1$ values ordered linearly in real time 
  $t_r$. Indirect time can be sampled in any order we wish.
- The two-dimensional signal is a sum of signals of the form
$$S_k(t_1,t_2; t_r) = A_k \cdot e^{2 \pi i \left(t_1f_{k,1}(t_r)+t_2f_{k,2}(t_r)\right) - t_1/\tau_k}$$
- The effects of frequency non-stationarity during a single $t_1$ sampling is negligible.

I've recreated the three situations from *"Fast time-resolved NMR with non-uniform sampling"* for a 2D complex signal and spectrum.
In each case there were 1024 $\times$ 1024 sampling points, with $\Delta t_1 = \Delta t_2 = \frac{1}{512}\text{s}$ and the same section of spectrum near the resonant peak have been shown in each case. Noise was not added to any of these signals.

Two different modes of sampling was used. 
**In the first**, indirect time changes linearly with the real time ($t_2 \propto t_r$, linear schedule), i.e. the order of sampling is $T_{\propto} = (0, \Delta t_2, 2\Delta t_2 \dots)$. 
**In the second**, the order of sampling is a random permutation of the $T_{\propto}$, denoted $t_2 \perp t_r$ (shuffled schedule). 
The permutation is drawn from a uniform distribution of 1024-elements permutations.

#### Non-stationarity of direct time frequency $f_1$

The direct time frequency $f_1$ changes with $t_r$, while the indirect time frequency $f_2$ is stationary. Simulated signal was of the form
$$S(t_1,t_2;t_r) = e^{2\pi i ((50+10\frac{t_r}{1024})t_1+100t_2)-t_1/256} $$

Linear schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/direct_lin.png)
Schuffled schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/direct_perp.png)
The indirect time frequency $f_2$ changes with $t_r$, while the direct time frequency $f_1$ is stationary. Simulated signal was of the form
$$S(t_1,t_2;t_r) = e^{2\pi i (50t_1+(100+2.5\frac{t_r}{1024})t_2)-t_1/256} $$

Linear schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/indirect_lin.png)
Schuffled schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/indirect_perp.png)
Both frequencies change with $t_r$
$$S(t_1,t_2;t_r) = e^{2\pi i ((50+10\frac{t_r}{1024})t_1+(100+2.5\frac{t_r}{1024})t_2)-t_1/256} $$

Linear schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/both_lin.png)
Schuffled schedule:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/both_perp.png)

Apart from the differences coming from lack of symmetry compared to the signal used in *"Fast time-resolved NMR with non-uniform sampling"*, I have noticed a much bigger range of artifacts in spectrum coming from non-stationarity of $f_2$  than $f_1$ in the mode $t_2 \perp t_r$ . For a slightly larger non-stationarity, than presented above, the artifacts completely dominate. Here with $f_1(t_r)=\text{const} = 50, \ f_2(t_r) = 100 + 10t_r$ :
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/indirect_perp_big.png)
## Examples of Compressed Sensing reconstruction
I've implemented a very simple algorithm for 1D/2D CS reconstruction, using embedded conic solver from CVXPY library. Optimization problem is defined as $$\underset{\tilde S_r}{\text{argmin}} \lVert \tilde{S_r} \rVert_1 \ ; \quad \text{mat}(\mathcal{M}\mathcal{F}^{-1}) \cdot \text{vec} (\tilde S_r ) - \text{vec} (S) \leq \delta$$
Where $\tilde{S_r}$ is the searched for spectrum, $\text{vec}(\cdot)$ is a vectorization transformation $v^{di +j} = m_{\ j}^i$ for $c\times d$  m matrix or identity for m vector, $\text{mat}(\cdot)$ is the matricization transformation $m_{\ \ dk+l}^{bi+j} = t_{\ \ kl}^{ij}$ for $a\times b \times c \times d$ tensor or identity for t matrix, $\mathcal{F}$ is a Fourier transform operator (rank 4 tensor or matrix) and $\mathcal{M}$ is a sampling operator, which projects the full signal onto the subspace of sampled points of $S$.

The basic convex optimization handles small-sized problems well, but is inefficient for larger signals.

In the following two examples, spectrum has been reconstructed from a 10% subsample of datapoints.

Plot shows a real part of the spectrum of the original (up) and the reconstructed (down) signals. 
1D signal is of the form: $S(t) = e^{2\pi i \ 0.4 t - \frac{t}{256}} + \frac{1}{2} e^{2\pi i \ 0.7 t - \frac{t}{256}}$ , with added complex gaussian noise $\sigma = 0.3$ , 512 sampling points and sampling rate $\Delta t = \frac {1}{512} \text{s}$

The original signal:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/1d_rec_orig.png)
The reconstruction:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/1d_rec_rec.png)
2D signal is of the form: $S(t) = e^{2\pi i (0.1 t_1 + 0.2 t_2) - \frac{t}{20}} + \frac{1}{2} e^{2\pi i (0.3 t_1 + 0.6 t_2) - \frac{t}{20}}+ \frac{1}{4} e^{2\pi i (0.5 t_1 + 0.5 t_2) - \frac{t}{20}}$ , with added complex gaussian noise $\sigma = 0.2$ , $40\times 40$ sampling points and sampling rate $\Delta t_1 = \Delta t_2 = \frac {1}{40} \text{s}$

The original signal:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/2d_rec_orig1.png)
The reconstruction:
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/2d_rec_rec1.png)
## Non - uniform sampling vs non-stationarity compromise

For non-uniformly sampled signals with non-stationary frequency, both the degree of non-stationarity and the loss of information in NUS create artifacts in the resulting spectrum.
If we assume, that the real time difference between the sampling of consecutive data points (or of consecutive 1D spectra in 2D case) is constant, then 
the shuffled schedule with non-uniform sampling may allow us to reconstruct the whole signal, with "slowed down" real time, thus reducing frequency non-stationarity.
Lowering the number of sampled points, however, itself introduces more artifacts because of the data loss.
I have simulated the effect of changing the ratio of sampled points on the reconstructed spectrum.

For a signal $S(t_1;t_r) = 2e^{2\pi i (0.2 + \theta \frac{t_r}{512})t_1 - \frac{t_r}{170}} + e^{2\pi i (0.5 + \theta \frac{t_r}{512})t_1 - \frac{t_r}{170}}$ with 512 sampling points, $\Delta t_1 = \Delta t_r = \frac{1}{512}$ plus gaussian noise $\sigma = 0.3$. The gifs show results of spectrum reconstruction for increasing sampling ratios $10\%, \ 20 \%, \cdots, 100\%$ and for three different values of frequency rate of change $\theta$. Schedule $t_1 \perp t_r$ was used. Lower sampling ratio means that less real time have passed during signal acquisition.
There is a compromise between NUS and non-stationarity. Faster change of frequency lowers the optimal ratio, while bigger amount of noise increases it. 

Simulation with $\theta = 0.01$ : 
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/1d_compromise_1.gif)
Simulation with $\theta = 0.02$ : 
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/1d_compromise_2.gif)
Simulation with $\theta = 0.03$ : 
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/1d_compromise_3_1.gif)
Similar simulation was performed with 2D spectra.
$S(t_1,t_2; t_r) = 2e^{2 \pi i ((0.2 + \theta \frac{t_r}{40})t_1 + (0.4 + \theta \frac{t_r}{40})t_2) - \frac{t_1}{40}} + e^{2 \pi i ((0.6 + \theta \frac{t_r}{40})t_1 + (0.3 + \theta \frac{t_r}{40})t_2) - \frac{t_1}{40}}$
$40\times40$ sampling points, $\Delta t_1 = \Delta t_2 = \Delta t_r = \frac{1}{40}$, plus gaussian noise $\sigma = 0.3$ .
In 2D spectrum I have again observed, that the effects of non-stationarity strongly dominated and very low sampling ratios was favorable.

Simulation with $\theta = 0.02$ :
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/2d_compromise_2.gif)
Simulation with $\theta = 0.03$ :
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/2d_compromise_3.gif)
## Time-resolved non-uniform sampling

The combination of NUS and shuffled schedule of signal acquisition can be used in 2D spectroscopy to investigate time dependence of resonant frequencies during a process happening in real-time, like a change of temperature or an ongoing chemical reaction.
First, during of the process, 1D spectra with randomly selected values of indirect time $t_2$ are acquired for as long as the process goes. Its number depends on the duration of the process in real time and the interval between consecutive acquisitions.
The spectra can then be grouped and formed into multiple non-uniformly sampled 2D spectra with schedule $t_2 \perp t_r$ . Size of the groups can be adjusted, to allow for the best NUS - non-stationarity compromise. The undersampled 2D spectra are reconstructed with Compressed Sensing and a time dependence of resonant frequencies can be observed.

A signal of the form $S(t_1,t_2;t_r) = e^{2 \pi i ((0.20+0.025 t_r/n)t_1 + (0.4+0.015t_r/n)t_2)}$ (without the decay or gaussian noise) with $n=40$ , $\Delta t_1 = \Delta t_2 = \Delta t_r = \frac 1 n$ was simulated. 250 1D spectra with random $t_2$'s have been obtained and grouped into 30 2D spectra, with a sampling ratio of 20% for each.
Each 2D spectrum was reconstructed using CS and plotted. The image shows, how the resonant peak shifts from $(8 \ \text{Hz},16 \ \text{Hz})$ to $(14.25 \ \text{Hz}, 19.75 \ \text{Hz})$ in time range $t_r \in [0,250 \Delta t_r]$. The artifacts, that can be attributed to non-stationarity are still visible.
![a](https://github.com/w-smialek/NMR-signal-analysis/blob/main/2d_TS1.gif)
