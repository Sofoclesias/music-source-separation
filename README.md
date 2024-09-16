# music-source-separation
Repositorio para modelos enfocados en la separación de pistas instrumentales en canciones.

Al importar las dependencias, se debe agregar .windows. entre signal y la técnica de similitud en 'nussl/core/constants.py'

* WINDOW_HAMMING = scipy.signal.hamming.__name__
* WINDOW_HANN = scipy.signal.hann.__name__ 
* WINDOW_BLACKMAN = scipy.signal.blackman.__name__