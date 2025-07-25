## Filtering by smoothness of the plasma current to find the permanent state
`permanent_state_filtering.py` file

two filter:

1. `scipy.lfilter`: unidirectional lowpass filter. Problem: it introduces a delay. However, the delay is mathematicaly computable as $(ntaps-1)/2$ samples on the right with $ntaps$, the number of coefficient of the filter.

2. `scipy.filtfilt`: bidirectional filter with no delay. As it is bidirectional, the computational time is twice longer.