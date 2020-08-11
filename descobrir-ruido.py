import pydicom
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


def modelo_linear(x, results):
	y = 0
	params = list(results.params)

	# O vetor x tem 1 elemento a menos. Portanto, devemos somar o coeficiente livre da função após o looping
	for i in range(len(x)):
		y += x[i]*params[i+1]

	# Somando o coeficiente livre
	y += params[0]

	return y

def scaling(vet):
	#vet = (vet-np.amin(vet))/(np.amax(vet)-np.amin(vet)) * 255#65535
	vet = vet * (254 / np.amax(vet))
	return vet


realizacoes = []
qtdRealizacoes = 10

for i in range(qtdRealizacoes):
	i = i+1
	ds = pydicom.read_file("tentativa"+str(i)+"/_7.dcm")

	# Load dimensions based on the number of rows and columns
	ConstPixelDims = (int(ds.Rows), int(ds.Columns))
	#print("Quantidade de Rows w Cols")
	print(ConstPixelDims)

	# Load spacing values (in mm)
	ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
	#print("Valor do Pixel Spacing")
	print(ConstPixelSpacing)



	# Lista começando em 0, finalizando em
	x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
	#x = 2048
	y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
	#y = 1792


	# The array is sized based on 'ConstPixelDims'
	ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

	ArrayDicom[:, :] = ds.pixel_array  
	ArrayDicom = ArrayDicom.astype(np.float32)
	ArrayDicom = ArrayDicom[0:300, 700:1250]
	#ArrayDicom = scaling(ArrayDicom)

	#ArrayDicom = scaling(ArrayDicom)
	#ArrayDicom = ArrayDicom.astype('uint8')

	# printa algumas infs da realização
	media = np.mean(ArrayDicom)
	std = np.std(ArrayDicom)
	amax = np.amax(ArrayDicom)
	amin = np.amin(ArrayDicom)

	print("Valor Max: " + str(amax))
	print("Valor Min: " + str(amin))
	print("Média: " + str(media))
	print("Std: " + str(std))

	print("\n")

	# Empilha todas as realizações
	realizacoes.append(ArrayDicom)

	# Salva a imagem em PNG
	#img = Image.fromarray(ArrayDicom)
	#img.save('test.png')

	
print(realizacoes)

# calcula a média na terceira dimensão
media_terceira_dimensao = np.mean(realizacoes, axis=0)
variancia_terceira_dimensao = np.var(realizacoes, axis=0)

array_media = np.asarray(media_terceira_dimensao).flatten()
array_variancia = np.asarray(variancia_terceira_dimensao).flatten()

# Platando o array_media no eixo x e a array_variancia no eixo y


#exit()
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(array_media, array_variancia, 'o', label="data", markersize=1)
#ax.scatter(array_media, array_variancia, s = 1)

#ax.set(xlabel='Médias', ylabel='Variâncias',
#	   title='Médias x Variâncias')

#ax.grid()

#fig.savefig("mediasxvariancia.png")



#"""""""""""""""""
# Regressão Linear
#"""""""""""""""""




print(len(array_media))
print(len(array_variancia))


X = np.column_stack((np.ones(len(array_media)),
					array_media	
))

print(X)


model = sm.OLS(array_variancia, X)
results = model.fit()

print(results.summary())


y = []
for i in range(len(array_media)):

	x = [
		array_media[i]
		]
	#predicao = modelo_linear(x, results)

	# Calcula PREÇO REAL - PREÇO PREDITO
	#diferenca_de_preco = row['preco'] - preco_predito
	#data_com_urls.at[index, 'diferenca_de_preco'] = diferenca_de_preco
	#print("Diferença de preço:")
	#print(diferenca_de_preco)

	y.append(modelo_linear(x, results))

#print(y)

ax.plot(array_media, y, 'r--.', label="OLS", markersize=1)
fig.savefig("mediaxvariancia.png")