import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def t_bar(self, x, ymin, ymax, segment_length, **kwargs):
    """
    Aggiunge una barra verticale con un segmento orizzontale all'estremo superiore
    a un oggetto Axes.
    
    Parametri:
    self: oggetto Axes
    x: posizione sulla x della barra verticale
    ymin: valore minimo sull'asse y della barra
    ymax: valore massimo sull'asse y della barra
    segment_length: lunghezza del segmento orizzontale
    **kwargs: ulteriori argomenti per definire le proprietà del grafico
    """
    # Plotta la barra verticale sull'asse fornito
    self.vlines(x=x, ymin=ymin, ymax=ymax, **kwargs)
    
    # Calcola le estremità del segmento orizzontale
    x_start = x - segment_length / 2
    x_end = x + segment_length / 2
    
    # Plotta il segmento orizzontale sull'asse fornito
    self.hlines(y=ymax, xmin=x_start, xmax=x_end, **kwargs)

# Aggiungi il metodo alla classe Axes
Axes.t_bar = t_bar
