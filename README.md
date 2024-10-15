# music-source-separation
Repositorio para modelos enfocados en la separación de pistas instrumentales en canciones.

Hola Gerardo xd

Al importar las dependencias, se debe agregar .windows. entre signal y la técnica de similitud en 'nussl/core/constants.py'

* WINDOW_HAMMING = scipy.signal.hamming.__name__
* WINDOW_HANN = scipy.signal.hann.__name__ 
* WINDOW_BLACKMAN = scipy.signal.blackman.__name__

Anexos:

https://drive.google.com/drive/folders/1Bc28KCx4fkmkDmRaUcBPbAFrdABzFcCI?usp=sharing