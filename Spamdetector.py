import streamlit as st
import pickle

model = pickle.load(open('spam123.pkl','rb'))
cv = pickle.load(open('vec123.pkl','rb'))

def main():
    st.title("Email Spam Classificatioln Application")
    st.write("This is A Machine Learning application to classify")
    st.subheader("Classification")
    user_input=st.text_area("Enter an Email to classify", height=150)
    if st.button("classify"):
        if user_input:
            data=[user_input]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success('This is a Ham message!', icon="✅😊👍")
            else:
                st.error("This a Spam email", icon="❌😢🙅‍♂️")
        else:
            st.write("Please Enter an Email to classiy")
main()
