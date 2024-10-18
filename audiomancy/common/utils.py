"""
Módulo adicional para descargar y descomprimir archivos.
"""

import requests
import zipfile
import tarfile
import os
import shutil
from mega import Mega
from tqdm import tqdm

def download_file(url: str, destination: str):
    """Descarga archivos que sean recuperables mediante requests.

    Args:
        url (str): Enlace de descarga.
        destination (str): Path de destino para el archivo descargado.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    with open(destination, 'wb') as file, tqdm(
        desc=destination, 
        total=total_size, 
        unit='iB', 
        unit_scale=True, 
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def download_mega(url: str, destination: str):
    """Descarga archivos de MEGA.

    Args:
        url (str): Enlace de descarga.
        destination (str): Path de destino para el archivo descargado.
    """
    mega = Mega()
    m = mega.login() 
    file = m.find(url)

    file_size = file['size']
    file_path = os.path.join(destination, file['name'])

    with open(file_path, 'wb') as f, tqdm(
        desc=file['name'],
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        file_data = m.download(file, file_path)
        for chunk in file_data:
            f.write(chunk)
            bar.update(len(chunk))

def unzip_file(zip_path: str, extract_to: str):
    """Descomprime archivos .zip. Si el nombre del archivo es igual
    a la primera carpeta interna, esta se vuelve la principal.

    Args:
        zip_path (str): Ubicación del archivo .zip.
        extract_to (str): Destino final.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_content = zip_ref.namelist()

        first_folder = zip_content[0].split('/')[0]
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]

        if first_folder == zip_name:
            temp_extract_path = os.path.join(extract_to, 'temp_extract')
            zip_ref.extractall(temp_extract_path)
            source_path = os.path.join(temp_extract_path, first_folder)
            for file_name in os.listdir(source_path):
                shutil.move(os.path.join(source_path, file_name), extract_to)
            shutil.rmtree(temp_extract_path)
        else:
            zip_ref.extractall(extract_to)

def untar_file(tar_path: str, extract_to: str):
    """Descomprime archivos .tar.gz. Si el nombre del archivo es igual
    a la primera carpeta interna, esta se vuelve la principal.

    Args:
        tar_path (str): Ubicación del archivo .zip.
        extract_to (str): Destino final.
    """
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_content = tar_ref.getnames()

        first_folder = tar_content[0].split('/')[0]
        tar_name = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]

        if first_folder == tar_name:
            temp_extract_path = os.path.join(extract_to, 'temp_extract')
            tar_ref.extractall(temp_extract_path)
            source_path = os.path.join(temp_extract_path, first_folder)
            for file_name in os.listdir(source_path):
                shutil.move(os.path.join(source_path, file_name), extract_to)
            shutil.rmtree(temp_extract_path)
        else:
            tar_ref.extractall(extract_to)