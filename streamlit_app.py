import streamlit as st
import cv2 
from PIL import Image
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras



@st.cache
def load_image(img):
    im = Image.open(img)
    #Model of mask detection
    #model_path = "./modele/model_mask.h5"
    #model = keras.models.load_model(model_path)

    return im

#Loading models
cascade_path =  "./cascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

#Model of mask detection
model_path = "./modele/model_mask.h5"
model = keras.models.load_model(model_path)


def detect_faces(image):
    color = (0, 110, 127) #La couleur du carré qui entoure le visage détecté
    src = np.array(image.convert('RGB'))
    colored = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img_shape = (224, 224)

    #Detect face
    rect = cascade.detectMultiScale(colored)

    #Creation of dataframe
    df_info = pd.DataFrame(columns = ['Person', 'Date', 'Hour', 'Mask'])
    

    #Draw rectangle
    if len(rect) > 0:
        #Labels dictionnary with or without mask
        mask_label_dict = {'With mask' : 0, 'Without mask' : 1}
        
        count_mask = 0

        for i,[x, y, w, h] in enumerate(rect):
            #Trimming images
            img_trimmed = src[y : y + h, x : x + w]

            #Resizing images
            img_trimmed = cv2.resize(img_trimmed, img_shape)
            img_trimmed = np.expand_dims(np.array(img_trimmed), axis = 0) / 255

            #prediction of the model
            prob = model.predict(img_trimmed)
            prediction = prob.argmax(axis = -1)

            #Count the number of mask detected
            if prediction == 0:
                count_mask = count_mask + 1 
            
            #Check the label
            label = None
            for k, val in mask_label_dict.items(): 
                if prediction == val: 
                    label = k

            #drawing in the image
            cv2.rectangle(src, (x, y), (x+w, y+h), color)
            cv2.rectangle(src, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(src, f"Person{i+1} : {label}", (x + 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, .5, (0,0,0))


            #add rows in dataframe
            df_info = df_info.append({'Person': f"Person {i+1}",'Date':str(datetime.date.today()), 'Hour':str(datetime.datetime.today().time()), 'Mask': f"{label}" }, ignore_index=True)
    
    return src, rect, df_info, count_mask


def main():
    """ Face detection app"""

    st.title('Mask Detection App')
    st.text('Build with streamlit and OpenCV')

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select activity", activities) 
    
    if choice == 'Detection' :
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type = ['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.markdown("Original Image")
            st.image(our_image)
        
            #Face detection            
            if st.button("Process"):
                result_img, result_face, df, count = detect_faces(our_image)
                #Export dataframe in excel
                df.to_excel("output.xlsx")

                #Modification in streamlit
                st.subheader("Image with face(s) detected")
                st.image(result_img)
                if len(result_face) > 1 :
                    st.markdown(f"{len(result_face)} faces and {count} masks were found")
                else :
                    st.markdown(f"{len(result_face)} face and {count} mask were found")
                #st.success(f"Found {len(result_face)} faces")
                st.subheader("Details of person detected")
                st.dataframe(df)


    elif choice == 'About':
        st.subheader('About')
    


if __name__ == '__main__':
    main()