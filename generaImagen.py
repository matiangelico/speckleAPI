import matplotlib.pyplot as plt
from io import BytesIO
import base64

def colorMap(matriz): 
    
    imagen = plt.imshow(matriz, cmap='jet')
    plt.colorbar(imagen) 
    #plt.show()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Convertir la imagen a base64
    imagen64 = base64.b64encode(buf.read()).decode('utf-8')
    
    buf.close()
    plt.close()

    return imagen64