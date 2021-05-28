import streamlit as st
import keras, os, cv2, pickle
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.applications import VGG16
import cvlib as cv
from cvlib.object_detection import draw_bbox
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def load():
        
        # Specify canvas parameters in application
        st.title("WELCOME TO DrawMe !" )
        st.write('Choose between 15 classes:')
        st.write('Airplane, apple, banana, bicycle, car, cat, chair, duck, teddy bear, pizza, fire hydrant, train, elephant, knife, cup')
        bg_image = st.sidebar.file_uploader("Upload sketch:", type=["png", "jpg"])
                   
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 3)
        drawing_mode = st.sidebar.selectbox(
                "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
            )
        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        if bg_image:
            file_bytes = np.asarray(bytearray(bg_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
                
        else:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                update_streamlit=realtime_update,
                height=300,
                drawing_mode=drawing_mode,
                key="canvas",
                )
            if canvas_result.image_data is not None:
                    im = Image.fromarray(canvas_result.image_data.astype('uint8'), mode="RGBA")
                    im.save("test.png", "PNG")
                    im = Image.open('test.png')
                    background = Image.new("RGB", im.size, (255, 255, 255))
                    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
                    background.save('test.png', 'PNG', quality=80)
                    im = Image.open('test.png')
                    open_cv_image = np.array(im) 
                    opencv_image = open_cv_image[:, :, ::-1].copy()
                    
        if st.button('Detect Sketch'):
                    query=[]
                    import zipfile
                    with zipfile.ZipFile('final_class.zip', 'r') as zip_ref:
                            zip_ref.extractall()
                    with open('final_class.h5', 'rb') as file:
                        cnn = pickle.load(file)
                    vgg = VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))                
                    image = opencv_image
                    image2 = opencv_image
                    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    _,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)

                    # Find contours
                    cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cropped=[]
                    for i in range(len(cnts[0])):
                        c = cnts[0][i]
                        # Obtain outer coordinates
                        left = tuple(c[c[:, :, 0].argmin()][0])
                        right = tuple(c[c[:, :, 0].argmax()][0])
                        top = tuple(c[c[:, :, 1].argmin()][0])
                        bottom = tuple(c[c[:, :, 1].argmax()][0])
                        
                        #Draw bounding box using the extra coordinates
                        
                        cv2.rectangle(image2,(left[0]-10, top[1]-10), (right[0]+10, bottom[1]+10), (255,125,100), 2)
                        #Crop and save images using original coordinates
                        crop = image[top[1]:bottom[1], left[0]:right[0]]
                        cropped.append(crop)
                        
                    #Display bounded boxed sketch 
                    st.image(image2)
                    
                    #Class names in a string
                    string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
                    class_names=sorted(list(string1.split(",")))

                    for i in range(len(cnts[0])):
                        img = cropped[i]
                        st.image(img)
                        img = cv2.resize(img, (100,100))
                        input_img = np.expand_dims(img, axis=0)
                        
                        #Predict individual sketch features
                        input_img_feature = vgg.predict(input_img)
                        input_img_features = input_img_feature.reshape(input_img_feature.shape[0],-1)
                        
                        #Predict individual sketch class using SVM
                        prediction_SVM = cnn.predict(input_img_features)[0]
                        prediction=class_names[prediction_SVM]
                        query.append(prediction)
                        if 'duck' in query:
                            i=query.index('duck')
                            query[i]='bird'
                        classification = 'The image is classified as: ' +query[i].upper()
                        st.info(classification)
                    list1=[]
                    path='C:\\Users\\deepi\\Desktop\\heroku\\Combination'
                    for file in os.listdir(path):
                      list1.append(cv2.imread(os.path.join(path,file)))

                    
                    rank={}
                    label_dict={}
                    bbox_dict={}
                    conf_dict={}
                    
                    
                    def get_rank_query_label_difference(i,label1,flag):
                    #obtain count of query classes in label1
                        for j in query:
                            if j in label1:
                                flag+=1
                    #find difference to rank the photos    
                        if (len(query)-flag)==0:
                            rank[i]=1
                        elif (len(query)-flag)==1:
                            rank[i]=2
                        else:
                            rank[i]=3
                    rank={}
                    gif_runner = st.image('loading.gif')
                    
                    for i in range(0,len(list1)):
                        im = list1[i]
                        
                        #Perform object detection on photos stored in the list
                        bbox, label, conf = cv.detect_common_objects(im)
                        label1=[]
                        bbox1=[]
                        conf1=[]
                        flag=0
                        for param in label:
                            if param in query:
                                
                                #Read labels and compare with query to save only query labels
                                index=label.index(param)
                                label1.append(param)
                                bbox1.append(bbox[index])
                                conf1.append(conf[index])
                                label[index]="null"
                                                                                 
                        
                        if label1 !=[]:
                            
                            #Save the bounding box coordinates for the query labels 
                            label_dict["label{0}".format(i)]=label1
                            bbox_dict["bbox{0}".format(i)]=bbox1
                            conf_dict["conf{0}".format(i)]=conf1
                            
                            #Find ranking for every photo
                            get_rank_query_label_difference(i,label1,flag)            

                    #Sort photos in order of rank        
                    sort_orders = sorted(rank.items(), key=lambda x: x[1], reverse=False)
                    gif_runner.empty()
                    st.success("THE RESULTS OBTAINED ARE")
                    j=0
                    for i in sort_orders:
                        im=list1[i[0]]
                        
                        #Draw bounding box around the objects and display the output
                        output_image = draw_bbox(im, bbox_dict["bbox{0}".format(i[0])], label_dict["label{0}".format(i[0])], conf_dict["conf{0}".format(i[0])])
                        output = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
                        st.image(output)
                   



                            
