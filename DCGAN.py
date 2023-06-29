import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchinfo import summary
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import jovian
import torch.nn as nn
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import torch.nn.functional as F
import time
import cv2

project_name = 'dcgan'

# ## Dataset
# Como dataset utilizaremos un conjunto de imágenes de libre elección. Tendremos que especificar la carpeta
# que contiene las imágenes

DATA_DIR = './directorio'
print('\nTraining images directory: ')
print(os.listdir(DATA_DIR))

# Mostramos el nombre de las primeras 10 imágenes para confirmar que los datos se han leído correctamente
print('\nImages inside: ')
print(os.listdir(DATA_DIR + '/images')[:10])
print('\n')

# Vamos a cargar este conjunto de datos utilizando la clase `ImageFolder` de `torchvision`.
# También redimensionaremos y recortaremos las imágenes a 64x64 px,
# y normalizaremos los valores de los píxeles con una media y una desviación estándar de 0,5 para cada canal.
# Esto asegurará que los valores de los píxeles están en el rango `(-1, 1)`,
# que es más conveniente para el entrenamiento del discriminador.
# También crearemos un cargador de datos para cargar los datos por lotes.

image_size = 64
batch_size = 128
save_flag = 50
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)]))


train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)

# Vamos a crear funciones de ayuda para desnormalizar los tensores de imagen
# y mostrar algunas imágenes de muestra de un lote de entrenamiento.

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


jovian.commit(project=project_name, environment=None)
print('\n')


# ## Aprovechando la GPU
# Para utilizar sin problemas una GPU, si hay una disponible, definimos un par de funciones de ayuda
# (`get_default_device` & `to_device`) y una clase de ayuda `DeviceDataLoader`
# para mover nuestro modelo y datos a la GPU, si hay una disponible.

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# En función de dónde se esté ejecutando este Notebook, el dispositivo por defecto
# podría ser una CPU (`torch.device('cpu')`) o una GPU (`torch.device('cuda')`).

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)

# ## Red Discriminadora
#
# El discriminador toma una imagen como entrada e intenta clasificarla como "real" o "generada".
# En este sentido, es como cualquier otra red neuronal.
# Utilizaremos una red neuronal convolucional (CNN) que produce un único número de salida para cada imagen.

discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())

# Traslademos el modelo discriminador al dispositivo elegido.

discriminator = to_device(discriminator, device)

print('Discriminator info:\n')
print(discriminator)
print('\n\n\n')

summary(discriminator, input_size=(batch_size, 3, 64, 64))
print('\n\n\n')

# ## Red Generativa
#
# La entrada del generador suele ser un vector o una matriz de números aleatorios (denominado tensor latente)
# que se utiliza como semilla para generar una imagen. El generador convertirá un tensor latente de forma `(128, 1, 1)`
# en un tensor de imagen de forma `3 x 28 x 28`. Para conseguir esto, usaremos la capa `ConvTranspose2d` de PyTorch,
# que se conoce como convolución transpuesta (también conocida como deconvolución)

latent_size = 128

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

# Utilizamos la función de activación TanH para la capa de salida del generador.
#
# "La activación ReLU (Nair & Hinton, 2010) se utiliza en el generador con la excepción de la capa de salida,
# que utiliza la función TanH. Observamos que el uso de una activación acotada permitía al modelo aprender
# más rápidamente a saturar y cubrir el espacio de color de la distribución de entrenamiento.
# Dentro del discriminador encontramos que la activación rectificada con fugas funciona bien,
# especialmente para el modelado de mayor resolución."
#
# Nótese que como las salidas de la activación TanH se encuentran en el rango `[-1,1]`,
# hemos aplicado la transformación similar a las imágenes del conjunto de datos de entrenamiento.
# Vamos a generar algunas salidas (opcionalmente) utilizando el generador y visualizarlas como imágenes
# transformando y desnormalizando la salida.

xb = torch.randn(batch_size, latent_size, 1, 1)  # random latent tensors
fake_images = generator(xb)

# Como era de esperar, la salida del generador es básicamente ruido aleatorio, ya que aún no lo hemos entrenado.
# Movamos el generador al dispositivo elegido.

generator = to_device(generator, device)

print('\n\n\nGenerator info:\n')
print(generator)
print('\n\n\n')

summary(generator, input_size=(batch_size, latent_size, 1, 1))
print(discriminator)
print('\n\n\n')


# ## Entrenamiento del Discriminador

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


# ## Entrenamiento del Generador

def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


# Vamos a crear un directorio donde podamos guardar las salidas intermedias del generador para
# inspeccionar visualmente el progreso del modelo.
# También crearemos una función de ayuda para exportar las imágenes generadas.

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# Utilizaremos un conjunto fijo de vectores de entrada al generador para ver cómo las imágenes individuales
# generadas evolucionan con el tiempo a medida que entrenamos el modelo.
# Vamos a guardar un conjunto de imágenes antes de empezar a entrenar nuestro modelo (opcionalmente).

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

jovian.commit(project=project_name, environment=None)


# ## Bucle de entrenamiento

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers (Adam)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        if epoch % save_flag == 0:
            # Save generated images
            save_samples(epoch + start_idx, fixed_latent, show=False)
            # Save the model checkpoints
            torch.save(generator.state_dict(), 'saves/G/G-%s.pth' % epoch)
            torch.save(discriminator.state_dict(), 'saves/D/D-%s.pth' % epoch)

    return losses_g, losses_d, real_scores, fake_scores


# Ya estamos listos para entrenar el modelo. Se pueden probar distintos valores de learning rate y epochs.

lr = 0.00005
epochs = 7500

jovian.reset()
jovian.log_hyperparams(lr=lr, epochs=epochs)
print('\n\n\n')

inicio = time.time()  # Empezamos a medir el tiempo de ejecución

history = fit(epochs, lr)

fin = time.time()  # Terminamos de medir el tiempo de ejecución y lo mostramos
number = fin - inicio
print('\n\n\nExecution time: ')
print(number)
print('\n')

with open('time.txt', 'w') as f:
    f.write('%f' % number)

losses_g, losses_d, real_scores, fake_scores = history

jovian.log_metrics(loss_g=losses_g[-1],
                   loss_d=losses_d[-1],
                   real_score=real_scores[-1],
                   fake_score=fake_scores[-1])

# Ahora que hemos entrenado los modelos, podemos guardar los puntos de control finales.
# Save the model checkpoints
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

# Podemos visualizar el proceso de entrenamiento combinando las imágenes de muestra generadas
# después de cada 50 epochs, en este caso, en un vídeo utilizando OpenCV. Podemos cambiar el intervalo con el
# que se generan las imágenes modificando la variable 'save_flag'

vid_fname = 'dcgan_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'MP4V'), 1, (530, 530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()

# También podemos visualizar cómo cambia la pérdida con el tiempo. Visualizar las pérdidas
# es muy útil para depurar el proceso de entrenamiento.

plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');
plt.savefig('Losses.png')

plt.clf()
plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');
plt.savefig('Scores.png')

# ## Guardado y Commit
# Finalmente, procedemos a guardar todos los datos de este Notebook en Jovian:

jovian.commit(project=project_name,
              outputs=['G.pth', 'D.pth', 'dcgan_training.avi'],
              environment=None)
