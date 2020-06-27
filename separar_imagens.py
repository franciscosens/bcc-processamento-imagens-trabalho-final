import os
import random
from shutil import copyfile

try:
    from cv2 import cv2 as cv2
except ImportError:
    pass

DATADIR = "C:\\Users\\franc\\Desktop\\deep-learning\\my-deep-learning-second-project\\dataset"
DATADIR_TRAIN = DATADIR + "\\train"
DATADIR_VAL = DATADIR + "\\val"


def separar_imagens():
    if not os.path.exists(DATADIR_TRAIN):
        os.mkdir(DATADIR_TRAIN)

    if not os.path.exists(DATADIR_VAL):
        os.mkdir(DATADIR_VAL)

    arquivos = os.listdir(DATADIR)
    imagens_para_separar = []
    for image in arquivos:
        if os.path.isfile(os.path.join(DATADIR, image)):
            imagens_para_separar.append(image)

    imagens_para_separar.sort()
    random.shuffle(imagens_para_separar)
    split_1 = int(0.8 * len(imagens_para_separar))
    split_2 = len(imagens_para_separar)

    imagens_train = imagens_para_separar[:split_1]
    imagens_val = imagens_para_separar[split_1:split_2]

    copiar_imagens(imagens_train, DATADIR_TRAIN)
    copiar_imagens(imagens_val, DATADIR_VAL)


def copiar_imagens(imagens, path_destination):
    imagens_csv = ""
    linhas_csv = []
    with open(DATADIR + '\\annotation.csv', "r") as csvfile:
        linhas = csvfile.readlines()
        for linha in linhas:
            nome_arquivo = linha.split(',')[0]
            linha = {
                "nome": nome_arquivo,
                "linha": linha
            }
            linhas_csv.append(linha)

    for image in imagens:
        path_destination_aux = path_destination + "\\" + image
        path = DATADIR + "\\" + image
        copyfile(path, path_destination_aux)

        for line in linhas_csv:
            if image == line['nome']:
                if "\n" in line["linha"]:
                    imagens_csv = line["linha"] + imagens_csv
                else:
                    imagens_csv = line["linha"] + "\n" + imagens_csv

    imagens_csv = imagens_csv + "\n"
    f = open(path_destination + "\\" + "annotation.csv", "w")
    f.write(imagens_csv)
    f.close()


separar_imagens()
