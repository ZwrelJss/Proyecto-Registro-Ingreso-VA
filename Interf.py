import tkinter as tk
import csv
import numpy as np
from skimage.io import imread_collection , concatenate_images
from tkinter import ttk
from PIL import ImageTk, Image
import time
import os
from tkinter import messagebox
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io, transform
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import absl.logging
#Onehot Encoding the labels.
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical


class Aplicacion(tk.Tk):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("Registro de ingreso")
		#self.config(width = 500, height = 500)
		self.geometry("500x500+400+50")
		self.resizable(0, 0)


		self.estadoLectura = 0
		self.video_capture = cv2.VideoCapture(0)
		icoN = cv2.imread("leerU.png")
		#cv2.imshow("",icoN)

		self.et = ttk.Label(self, text="Registro de Ingreso")
		self.et.place(x=200, y=20)
		self.et.place()

		self.boton_Iniciar = ttk.Button(self, text="Iniciar", command=self.iniciar)
		self.boton_Iniciar.place(x=20, y=470)
		self.boton_Iniciar.place()

		self.boton_Registrar = ttk.Button(self, text="Registrar", command=self.registrar)
		self.boton_Registrar.place(x=200, y=470)
		self.boton_Registrar.place()

		self.boton_Actualizar = ttk.Button(self, text="Actualizar", command=self.actualizar)
		self.boton_Actualizar.place(x=400, y=470)
		self.boton_Actualizar.place()

		while True:
			#Cargando el modelo
			modelo = load_model('CNN/model-019.model')
			face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			labels_dict = os.listdir('./Data')
			color_dict = 'red'

			# Accediendo al registro
			registros = []
			with open('./Registro.csv') as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				for student in csv_reader:
					registros.append(student[0])
					#print(student[0])

			if self.estadoLectura == 1:
				ret, frame = self.video_capture.read()
				gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				faces=face_clsfr.detectMultiScale(gray,1.3,5)
				for (x,y,w,h) in faces:
					face_img=gray[y:y+w,x:x+w]
					resized=resize(face_img,(len(face_img),64,64,3))

					result=modelo.predict(resized)
					label=np.argmax(result,axis=1)[0]
				
					cv2.putText(frame,'{}'.format(label),(x,y-50),1,1.3,(86,155,35),1,cv2.LINE_AA)
					# print(result[0][1])
					cv2.putText(frame,'{}'.format(labels_dict[label]),(x,y-25),2,1.1,(0,0,255),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

					with open('./RegistroIngresos.csv', mode='a', newline='') as archivo_csv:
					    # Crear un objeto escritor CSV
					    escritor_csv = csv.writer(archivo_csv)
					    fila = [labels_dict[label]]
					    # Escribir los datos en el archivo CSV
					    escritor_csv.writerow(fila)
					    
					
					if labels_dict[label] in registros:
					
						self.estadoLectura = 0
						#self.video_capture.release()
						#self.video_capture = cv2.VideoCapture(0)
						self.boton_Iniciar.configure(state="enabled")
						self.boton_Registrar.configure(state="enabled")
						self.boton_Actualizar.configure(state="enabled")
						messagebox.showinfo("Mensaje", "Ingreso exitoso para "+labels_dict[label])

					

			else:
				frame = icoN

			#root = tk.Tk()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			frame_resized = cv2.resize(frame, (400, 400))
        
			# Crear una imagen PIL desde el marco redimensionado
			img = Image.fromarray(frame_resized)

			# Crear un objeto ImageTk
			img_tk = ImageTk.PhotoImage(img)

			self.label = ttk.Label(self, image=img_tk)
			self.label.place(x=40, y=40)
			self.label.place()

			self.update()
		
			if cv2.waitKey(1) == ord('q'):
				break

	def iniciar(self):
		self.estadoLectura = 1
		self.boton_Iniciar.configure(state="disabled")
		self.boton_Registrar.configure(state="disabled")
		self.boton_Actualizar.configure(state="disabled")
	def actualizar(self):
		self.estadoLectura = 0
		self.boton_Iniciar.configure(state="disabled")
		self.boton_Registrar.configure(state="disabled")
		self.boton_Actualizar.configure(state="disabled")
		self.entrenamiento()

	def registrar(self):
		self.estadoLectura = 0
		self.boton_Iniciar.configure(state="enable")
		if not Registro.en_uso:
			self.ventana_secundaria = Registro()

	def entrenamiento(self):
		#path
		contenido = os.listdir('./Data')
		print(contenido)
		#cargar imagenes
		carpetas = []
		for i in range(len(contenido)):
			direccion = "./Data/"+contenido[i]+"/*.jpg"
			print(direccion)
			carpetas.append(imread_collection(direccion))
		#len cantidad de imagenes
		longitudes = []
		for i in range(len(carpetas)):
			nimgs = len(carpetas[i])
			print(nimgs)
			longitudes.append(nimgs)
		#union data
		images = []
		for i in range(len(carpetas)):
			for j in range(longitudes[i]):
				images.append(carpetas[i][j])

		print("Total de imagenes", len(images))	

		#plot the first image in the dataset
		#plt.imshow(images[0])
		#plt.show()
		#print(images[0].shape)

		Create_Y = []
		est = 0
		for i in range(len(longitudes)):
			if est == 0:
				Create_Y += [0]*longitudes[i]
				est = 1
			elif est == 1:
				Create_Y += [1]*longitudes[i]
				est = 0
		#print(Create_Y)
		Y = np.array(Create_Y)
		X = []
		for i in range(len(images)):
			nueva_imagen = transform.resize(images[i], (64, 64, 3))
			X.append(nueva_imagen)
		X  = np.array(X)

		#plot the first image in the dataset
		#plt.imshow(X[0])
		#plt.show()
		
		#print(X[0].shape)
		#print(X.shape[1:])

		#
		# Creacion de la red convulocional
		#

		modelo=Sequential() #Varias capas apiladas entre ellas

		modelo.add(Conv2D(200,(3,3),input_shape=X.shape[1:]))
		modelo.add(Activation('relu'))
		modelo.add(MaxPooling2D(pool_size=(2,2)))
		#La primera capa de CNN seguida por las capas de Relu y MaxPooling

		modelo.add(Conv2D(100,(3,3)))
		modelo.add(Activation('relu'))
		modelo.add(MaxPooling2D(pool_size=(2,2)))
		#La segunda capa de convolución seguida por las capas de Relu y MaxPooling

		modelo.add(Conv2D(50,(3,3)))
		modelo.add(Activation('relu'))
		modelo.add(MaxPooling2D(pool_size=(2,2)))
		#La tercera capa de convolución seguida por las capas de Relu y MaxPooling

		modelo.add(Flatten()) #Imagen profunda la vamos a hacer plana, es decir solo una dimension,
		#va tener toda nuestra informacion de la cnn

		modelo.add(Dropout(0.5)) # Apagamos 50% de las neuronas cada paso, asi evitamos sobreajustar
		#(evitar un solo camino de entrenamiento)

		modelo.add(Dense(50,activation='relu'))
		#Capa Densa de 50 neuronas

		modelo.add(Dense(2,activation='softmax')) #La capa final softmax con dos salidas para dos categorías
		#Softmax nos indica que tanta probabilidad tiene cada clase, y por ende saber cual tiene la maxima probabilidad

		modelo.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


		#Ya que tenemos 2 clases debemos esperar que shape[1] de y_train,y_val y y_test cambie de 0 a 1
		Y=to_categorical(Y)

		#print(Y[0])
		#print(Y[len(Y)-1])

		absl.logging.set_verbosity(absl.logging.ERROR)
		from sklearn.model_selection import train_test_split
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
		checkpoint = ModelCheckpoint('CNN/model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
		history=modelo.fit(X_train,Y_train,epochs=20,callbacks=[checkpoint],validation_split=0.2) #,shuffle = True)

		print(history.history.keys())

		#
		# Visualización de desempeño
		#

		#from matplotlib import pyplot as plt
		"""
		plt.plot(history.history['loss'],'r',label='training loss')
		plt.plot(history.history['val_loss'],label='Test loss')
		plt.xlabel('# epochs')
		plt.ylabel('loss')
		plt.legend()
		plt.show()

		plt.plot(history.history['accuracy'],'r',label='training accuracy')
		plt.plot(history.history['val_accuracy'],label='test accuracy')
		plt.xlabel('# epochs')
		plt.ylabel('loss')
		plt.legend()
		plt.show()
		"""
		print(modelo.evaluate(X_test,Y_test))

		#Making prediction
		#train_data,test_data,train_target,test_target
		y_pred=np.argmax(modelo.predict(X_test), axis = 1)
		y_true=np.argmax(Y_test,axis=1)
		
		#Plotting the confusion matrix
		confusion_mtx=confusion_matrix(y_true,y_pred)
		print(confusion_mtx)
		
		class_names=contenido

		# Plotting non-normalized confusion matrix
		#plot_confusion_matrix(y_true, y_pred, classes = class_names,title = 'Confusion matrix, without normalization')
		
		from sklearn.metrics import accuracy_score
		acc_score = accuracy_score(y_true, y_pred)
		print('Accuracy Score = ', acc_score)

		self.boton_Iniciar.configure(state="enabled")
		self.boton_Registrar.configure(state="enabled")
		self.boton_Actualizar.configure(state="enabled")

		messagebox.showinfo("Mensaje", "Actualizacion completa")




class Registro(tk.Toplevel):
	en_uso = False
	def __init__(self,  *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("Nuevo registro al sistema")
		#self.config(width = 800, height = 500)
		self.geometry("800x500+200+50")
		self.focus()
		self.grab_set()
		self.resizable(0, 0)

		self.__class__.en_uso = True

		self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
		self.estadoLectura = 0
		self.numeroframes = 100
		self.video_capture = cv2.VideoCapture(0)
		self.icoN = cv2.imread("leerU.png")

		self.et = ttk.Label(self, text="Nuevo registro al sistema")
		self.et.place(x=200, y=20)

		self.boton_Iniciar = ttk.Button(self, text="Iniciar Captura", command=self.nuevoregistro)
		self.boton_Iniciar.place(x=20, y=470)

		self.boton_Listo = ttk.Button(self, text="Listo", command=self.listo)
		self.boton_Listo.place(x=200, y=470)

		self.et = ttk.Label(self, text="Numero de Cuenta")
		self.et.place(x=450, y=40)
		self.caja_id = ttk.Entry(self)
		self.caja_id.place(x=450, y=70, width=300)

		self.et = ttk.Label(self, text="Nombre/s")
		self.et.place(x=450, y=120)
		self.caja_nombre = ttk.Entry(self)
		self.caja_nombre.place(x=450, y=150, width=300)

		self.et = ttk.Label(self, text="Apellido Paterno")
		self.et.place(x=450, y=200)
		self.caja_apPat = ttk.Entry(self)
		self.caja_apPat.place(x=450, y=230, width=300)

		self.et = ttk.Label(self, text="Apellido Materno")
		self.et.place(x=450, y=280)
		self.caja_apMat = ttk.Entry(self)
		self.caja_apMat.place(x=450, y=310, width=300)

		cont = 0;
		while True:
			
			if self.estadoLectura == 1:
				ret, frame = self.video_capture.read()
				self.guardarF(cont, frame)
				cont += 1

			else:
				frame = self.icoN

			if cont == self.numeroframes:
				self.estadoLectura = 0
				cont = 0
				self.boton_Iniciar.configure(state="enabled")
				self.boton_Listo.configure(state="enabled")
				self.caja_id.delete(0, tk.END)
				self.caja_nombre.delete(0, tk.END)
				self.caja_apPat.delete(0, tk.END)
				self.caja_apMat.delete(0, tk.END)
				self.video_capture.release()
				self.video_capture = cv2.VideoCapture(0)

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			frame_resized = cv2.resize(frame, (400, 400))
        
			# Crear una imagen PIL desde el marco redimensionado
			img = Image.fromarray(frame_resized)

			# Crear un objeto ImageTk
			img_tk = ImageTk.PhotoImage(img)

			label = ttk.Label(self, image=img_tk)
			label.place(x=40, y=40)
			#label.config(image=img_tk)
			#label.image = img_tk
			


			self.update()
			
			if cv2.waitKey(1) == ord('q'):
				break
		
	def nuevoregistro(self):
		self.estadoLectura = 1
		self.boton_Iniciar.configure(state="disabled")
		self.boton_Listo.configure(state="disabled")

	def guardarF(self, cont, frame):
		det = 0
		idn = self.caja_id.get()
		if len(idn) > 0:
			print("Listo")
		else:
			print("Campo Vacio de id")
			det += 1
		try:
			if type(int(idn)) is int:
				print("Es entero")
		except ValueError:
			print("No es entero")
			det += 1

		cnombre = self.caja_nombre.get()
		if len(cnombre) > 0:
			print("Listo")
		else:
			print("Campo Vacio de nombre")
			det += 1

		cap = self.caja_apPat.get()
		if len(cap) > 0:
			print("Listo")
		else:
			print("Campo Vacio de Apelido Paterno")
			det += 1

		cam = self.caja_apMat.get()
		if len(cam) > 0:
			print("Listo")
		else:
			print("Campo Vacio de Apelido Materno")
			det += 1

		if det == 0:
			self.estadoLectura = 1
			cop = self.recortarRostro(frame, cont)
			frame = cop

			try:
			    os.mkdir("./Data/"+str(idn))
			    print("Se ha creado la carpeta", str(idn))
			except FileExistsError:
			    print("La carpeta", str(idn), "ya existe")


			nombre = "."+ "/Data/" + str(idn)+"/" + str(idn) + "_"+ str(cont)+".jpg"
			cv2.imwrite(nombre,  frame)
			cv2.waitKey(100)
			if cont == self.numeroframes-1:
				messagebox.showinfo("Mensaje", "Registro exitoso")
				with open('./Registro.csv', mode='a', newline='') as archivo_csv:
				    # Crear un objeto escritor CSV
				    escritor_csv = csv.writer(archivo_csv)
				    fila = [idn, cnombre, cap, cam]
				    # Escribir los datos en el archivo CSV
				    escritor_csv.writerow(fila)
		else:
			if cont == self.numeroframes-1:
				messagebox.showinfo("Error", "Revisa los campos")
	def listo(self):
		#self.video_capture.release()
		self.__class__.en_uso = False
		super().destroy()

	def recortarRostro(self, fr, count):
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		auxFrame = fr.copy()
		cop = fr.copy()

		faces = self.faceClassif.detectMultiScale(gray, 1.1, 4)

		for (x,y,w,h) in faces:
			cv2.rectangle(fr, (x,y),(x+w,y+h),(128,0,255),2)
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
			cop = rostro
		cv2.destroyAllWindows()
		return cop

app = Aplicacion()
ventana.mainloop()